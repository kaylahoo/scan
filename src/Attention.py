import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, num_channel, squeeze_ratio=1.0):
        super(SEModule, self).__init__()
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)
        self.num_channel = num_channel

        blocks = [nn.Linear(num_channel, int(num_channel * squeeze_ratio)),
                  nn.ReLU(),
                  nn.Linear(int(num_channel * squeeze_ratio), num_channel),
                  nn.Sigmoid()]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        ori = x
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel, 1, 1)
        x = ori * x
        return x


class AttNormalization(nn.Module):
    def __init__(self, in_channel, nClass=16, kama=10, orth_lambda=1e-3, eps=1e-7, reuse=False):
        super(AttNormalization, self).__init__()
        self.nClass = nClass
        self.kama = kama
        self.in_channel = in_channel
        self.orth_lambda = orth_lambda
        self.eps = eps

        self.xk = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1, stride=1)
        self.xq = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 8, kernel_size=1, stride=1)
        self.xv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1)
        self.x_mask_filters = nn.Parameter(torch.randn(nClass, in_channel, 1, 1), requires_grad=True)
        # self.alpha = nn.Parameter(torch.ones(size=(nClass, 1, 1)) * 0.1)
        self.sigma = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        b, c, h, w = x.size()
        n = self.nClass
        # xk: (b,c//8,h,w) ; xq: (b,c//8,h,w) ; xv: (b,c,h,w) ; x_mask: (b,n,h,w)
        xk = self.xk(x)
        xq = self.xq(x)
        xv = self.xv(x)
        x_mask = F.conv2d(x, self.x_mask_filters, stride=1)


        # mask_w: (1,n,c)
        mask_w = torch.reshape(self.x_mask_filters, [1, n, c])
        sym = torch.matmul(mask_w, mask_w.permute(0, 2, 1))
        len = torch.sqrt(torch.sum(mask_w ** 2, dim=-1, keepdim=True))  # (1,n,1)
        loss = sym / (torch.matmul(len, len.permute(0, 2, 1)) + self.eps)
        loss -= cuda(torch.eye(n))
        # orth_loss = self.orth_lambda * torch.sum(loss, dim=(0, 1, 2), keepdim=False)
        orth_loss = torch.sigmoid(torch.sum(loss, dim=(0, 1, 2), keepdim=False))

        # torch.multinomial 采样
        sampling_pos = torch.multinomial(torch.ones(1, h * w) * 0.5, n)
        sampling_pos = torch.unsqueeze(sampling_pos, dim=0).expand(b, c // 8, n)
        xk_reshaped = torch.reshape(xk, (b, c // 8, h * w))
        # fast_filters: (b,c//8,n)
        fast_filters = torch.gather(xk_reshaped, dim=2, index=cuda(sampling_pos))

        xq_reshaped = torch.reshape(xq, (b, c // 8, h * w))
        # fast_activations: (b,n,h*w)
        fast_activations = torch.matmul(fast_filters.permute(0, 2, 1), xq_reshaped)
        fast_activations = torch.reshape(fast_activations, (b, n, h, w))

        # alpha = tf.clip_by_value(alpha, 0, 1)
        self.alpha.data = torch.clamp(self.alpha, 0, 1)

        # layout: (b,n,h,w) , layout计算出nClass个语义分布
        layout = torch.softmax((self.alpha * fast_activations + x_mask) / self.kama, dim=1)
        layout_expand = layout.view(b, 1, n, h, w)
        # xv_expand = torch.tile(xv.view(b, c, 1, h, w), dims=[1, 1, n, 1, 1])
        xv_expand = xv.view(b, c, 1, h, w).repeat(1, 1, n, 1, 1)

        hot_area = xv_expand * layout_expand
        cnt = torch.sum(layout_expand, [-1, -2], keepdim=True) + self.eps
        xv_mean = torch.mean(hot_area, [-1, -2], keepdim=True) / cnt
        xv_std = torch.sqrt(torch.sum((hot_area - xv_mean) ** 2, [-1, -2], keepdim=True) / cnt)
        xn = torch.sum((xv_expand - xv_mean) / (xv_std + self.eps) * layout_expand, dim=2)
        x = x + self.sigma * xn

        return x, orth_loss


class ChanAttNorm(nn.Module):
    def __init__(self, in_channel, out_channel, nClass=16, orth_lambda=0.1, eps=1e-7, kama=1):
        super(ChanAttNorm, self).__init__()
        self.in_c = in_channel
        self.out_c = out_channel
        self.nClass = nClass
        self.orth_lambda = orth_lambda
        self.eps = eps
        self.kama = kama

        self.ksa = nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=1)
        self.kr = nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=1)
        self.kn = nn.Parameter(torch.randn(self.nClass, self.in_c, 1, 1), requires_grad=True)
        self.ko = nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=1)
        self.alpha = nn.Parameter(torch.ones(size=(1, self.nClass, 1)) * 0.1)
        self.sigma = nn.Parameter(torch.zeros([1]))

    def forward(self, x):

        b, c, h, w = x.size()
        n = self.nClass
        # xsa: (b,c/8,h,w) ; xr: (b,c/8,h,w) ; xn: (b,n,h,w) ; xo: (b,c,h,w)
        xsa = self.ksa(x)
        xr = self.kr(x)
        xn = F.conv2d(x, self.kn, stride=1)
        xo = self.ko(x)

        # L0 loss
        # kn: (n,c,1,1)
        mask_w = torch.reshape(self.kn.permute(2, 3, 0, 1), [1, n, c])
        # loss 计算余弦相似度
        sym = torch.matmul(mask_w, mask_w.permute(0, 2, 1))  # (1,n,n)
        norm = torch.sqrt(torch.sum(mask_w ** 2, dim=-1, keepdim=True))
        loss = sym / (torch.matmul(norm, norm.permute(0, 2, 1)) + self.eps)
        loss -= cuda(torch.eye(n))
        orth_loss = torch.sum(loss ** 2, dim=(0, 1, 2), keepdim=False)
        orth_loss = self.orth_lambda * torch.log(orth_loss + 1)

        xsa_reshaped = torch.reshape(xsa, (b, c, h * w))
        xr_reshaped = torch.reshape(xr, (b, c, h * w))
        xn_reshaped = torch.reshape(xn, (b, n, h * w))

        relation = torch.matmul(xn_reshaped, xr_reshaped.permute(0, 2, 1))  # (b,n,c)
        norm_xr = torch.sqrt(torch.sum(xr_reshaped ** 2, dim=-1, keepdim=True))  # (b,c,1)
        norm_xn = torch.sqrt(torch.sum(xn_reshaped ** 2, dim=-1, keepdim=True))  # (b,n,1)
        relation_norm = relation / (torch.matmul(norm_xn, norm_xr.permute(0, 2, 1)) + self.eps)


        rela = torch.matmul(xsa_reshaped, xsa_reshaped.permute(0, 2, 1))  # (b,c,c)
        factor = rela * cuda(torch.ones((c, c)) - torch.eye(c))
        regular = torch.softmax(torch.mean(factor, dim=1, keepdim=True), dim=-1)  # (b,1,c)
        regular = regular.expand(b, n, c)
        self.alpha.data = torch.clamp(self.alpha, 0, 1)
        relation_norm = (relation_norm + self.alpha * regular) / self.kama

        '''
        sampling_chan = torch.multinomial(factor, n)  # (b,n)
        sampling_chan = torch.unsqueeze(sampling_chan, dim=-1).expand(b, n, c)
        regular = torch.gather(rela, dim=1, index=sampling_chan)
        
        group = torch.max(relation_norm, dim=1, keepdim=True)  # (b,1,c)
        groups = []
        for i in range(n):
            groups.append(torch.eq(group.indices, i) + self.eps)
        att_score = torch.cat(groups, dim=1).view(b, n, c, 1, 1).expand(b, n, c, h, w)
        '''

        att_score = torch.softmax(relation_norm, dim=1).view(b, n, c, 1, 1).expand(b, n, c, h, w)  # (b,n,c)

        xo_expand = torch.unsqueeze(xo, dim=1).expand(b, n, c, h, w)
        hot_area = xo_expand * att_score
        cnt = torch.sum(att_score, [-1, -2, -3], keepdim=True) + self.eps
        fo_mean = torch.sum(hot_area, [-1, -2, -3], keepdim=True) / cnt
        fo_std = torch.sqrt(torch.sum((hot_area - fo_mean) ** 2, [-1, -2, -3], keepdim=True) / cnt)
        out = torch.sum((xo_expand - fo_mean) / (fo_std + self.eps) * att_score, dim=1)
        out = x + self.sigma * out

        return out, orth_loss


class PosAttNorm(nn.Module):
    def __init__(self, in_channel, out_channel, patch_size=1, nClass=16, kama=10, eps=1e-7, orth_lambda=1e-3):
        super(PosAttNorm, self).__init__()
        self.fc = in_channel
        self.xc = out_channel
        self.nClass = nClass
        self.p_size = patch_size
        self.kama = kama
        self.eps = eps
        self.orth_lambda = orth_lambda
        self.att_score = None
        self.sampling = None

        self.ksa = nn.Conv2d(in_channels=self.fc, out_channels=self.xc, kernel_size=1, stride=1)
        self.kr = nn.Conv2d(in_channels=self.xc, out_channels=self.xc, kernel_size=1, stride=1)
        self.kn = nn.Parameter(torch.randn(nClass, self.xc, 1, 1), requires_grad=True)
        self.ko = nn.Conv2d(in_channels=self.xc, out_channels=self.xc, kernel_size=1, stride=1)

        self.alpha = nn.Parameter(torch.ones(size=(1, nClass, 1, 1)) * 0.1, requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.sigma = nn.Parameter(torch.zeros([1]), requires_grad=True)

    def forward(self, x, f, mask, reuse=False):
        b, c, h, w = x.size()
        n = self.nClass

        if not reuse:
            # fsa: (b,c,h,w) ; xr: (b,c,h,w) ; xn: (b,n,h,w) ; xo: (b,c,h,w)
            fsa = self.ksa(f)
            xr = self.kr(x)
            xn = F.conv2d(x, self.kn)
            xo = self.ko(x)

            # L0 loss
            # kn: (n,c,1,1)
            mask_w = torch.reshape(self.kn.permute(2, 3, 0, 1), [1, n, c])
            # loss 计算余弦相似度
            sym = torch.matmul(mask_w, mask_w.permute(0, 2, 1))  # (1,n,n)
            norm = torch.sqrt(torch.sum(mask_w ** 2, dim=-1, keepdim=True))
            loss = sym / (torch.matmul(norm, norm.permute(0, 2, 1)) + self.eps)
            loss -= cuda(torch.eye(n))
            orth_loss = torch.sum(loss ** 2, dim=(0, 1, 2), keepdim=False)
            orth_loss = self.orth_lambda * torch.log(orth_loss + 1)

            # hole: 0
            mask_unfold = F.unfold(mask, self.p_size, dilation=1, stride=self.p_size, padding=0)  # (b,1*p*p,(h/4)**2)  p_size=4
            fsa_unfold = F.unfold(fsa, self.p_size, dilation=1, stride=self.p_size, padding=0)  # (b,c*p*p,(h/4)**2)
            xr_unfold = F.unfold(xr, self.p_size, dilation=1, stride=self.p_size, padding=0)  # (b,c*p*p,(h/4)**2)

            factor = torch.mean(mask_unfold, dim=1)  # (b,(h/4)**2)
            sampling_pos_f = index = torch.multinomial(factor, 2*n)  # (b,2n)
            sampling_pos_f = torch.unsqueeze(sampling_pos_f, dim=1).expand(b, c*self.p_size*self.p_size, 2*n)  # (b,c*p*p,2n)
            sampling_pos_f = cuda(sampling_pos_f)
            filters_pos_f = torch.gather(fsa_unfold, dim=-1, index=sampling_pos_f)  # (b,c*p*p,2n)
            norm_f = torch.sqrt(torch.sum(filters_pos_f ** 2, dim=1, keepdim=True))  # (b,1,2n)

            sampling_pos_x = torch.multinomial((1-factor), 2*n)
            sampling_pos_x = torch.unsqueeze(sampling_pos_x, dim=1).expand(b, c*self.p_size*self.p_size, 2*n)
            sampling_pos_x = cuda(sampling_pos_x)
            filters_pos_x = torch.gather(xr_unfold, dim=-1, index=sampling_pos_x)
            norm_x = torch.sqrt(torch.sum(filters_pos_x ** 2, dim=1, keepdim=True))

            relation = torch.matmul(filters_pos_f.permute(0, 2, 1), filters_pos_x)  # (b,2n,2n) row:fi和m个xj的关系
            relation_norm = relation / (torch.matmul(norm_f.permute(0, 2, 1), norm_x)+self.eps)
            factor = torch.max(relation_norm, dim=-1)
            factor = torch.softmax(factor.values, dim=-1)  # (b,2n) #使用sigmoid貌似并不是一个好主意
            sampling_pos = torch.multinomial(factor, n)  # (b,n)
            sampling_pos = torch.gather(index, dim=-1, index=sampling_pos)
            self.sampling = sampling_pos
            sampling_pos = torch.unsqueeze(sampling_pos, dim=1).expand(b, c*self.p_size*self.p_size, n)
            sampling_pos = cuda(sampling_pos)
            filters_pos = torch.gather(fsa_unfold, dim=-1, index=sampling_pos)  # (b,c*p*p,n)
            filters_pos = filters_pos.reshape(b, c, self.p_size, self.p_size, n).permute(0, 4, 1, 2, 3)  # (b,n,c,p,p)

            filters_pos_groups = list(torch.split(filters_pos, 1, dim=0))  # b*(1,n,c,p,p)
            xr_groups = list(torch.split(xr, 1, dim=0))  # b*(1,c,h,w)
            activations_pos_groups = []
            for i in range(len(filters_pos_groups)):
                filters_pos_groups[i] = torch.squeeze(filters_pos_groups[i], dim=0)
                activations_pos_groups.append(F.conv2d(xr_groups[i], filters_pos_groups[i], stride=1, padding=self.p_size//2))
            activations_pos = torch.squeeze(torch.cat(activations_pos_groups, dim=0))  # (b,n,h,w)
            self.alpha.data = torch.clamp(self.alpha, 0, 1)
            self.att_score = torch.softmax((self.alpha * activations_pos + xn) / self.kama, dim=1).view(b, 1, n, h, w)

        else:
            xo = x
            b, c, h, w = xo.size()
            xo_unfold = F.unfold(xo, self.p_size, dilation=1, stride=self.p_size, padding=0)  # (b,c*p*p,(h/4)**2)
            self.sampling = torch.unsqueeze(self.sampling, dim=1).expand(b, c*self.p_size*self.p_size, n)
            self.sampling = cuda(self.sampling)
            filters = torch.gather(xo_unfold, dim=-1, index=self.sampling)  # (b,c*p*p,n)
            filters = filters.reshape(b, c, self.p_size, self.p_size, n).permute(0, 1, 4, 2, 3)
            filters_groups = list(torch.split(filters, 1, dim=0))  # b*(1,c,n,p,p)
            att_score_groups = list(torch.split(torch.squeeze(self.att_score, dim=1), 1, dim=0))  # b*(1,n,h,w)
            acts = []
            for i in range(len(filters_groups)):
                filters_groups[i] = torch.squeeze(filters_groups[i], dim=0)
                acts.append(F.conv2d(att_score_groups[i], filters_groups[i], stride=1, padding=self.p_size//2))
            xo = torch.squeeze(torch.cat(acts, dim=0)).view(b, c, h, w)  # (b,c,h,w)
            xo = x * mask + xo * (1 - mask)

        xo_expand = torch.unsqueeze(xo, dim=2).expand(b, c, n, h, w)
        hot_area = xo_expand * self.att_score
        cnt = torch.sum(self.att_score, [-1, -2], keepdim=True) + self.eps
        xo_mean = torch.sum(hot_area, [-1, -2], keepdim=True) / cnt
        xo_std = torch.sqrt(torch.sum((hot_area - xo_mean) ** 2, [-1, -2], keepdim=True) / cnt)
        out = torch.sum((xo_expand - xo_mean) / (xo_std + self.eps) * self.att_score, dim=2)
        out = x + self.sigma * out

        if not reuse:
            return out, orth_loss
        else:
            return out


def cuda(x, use_gpu=True):
    if use_gpu:
        return x.cuda()
    else:
        return x

