import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gradient(nn.Module):

    def __init__(self):
        super(Gradient, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.size()
        xs = torch.split(x, split_size_or_sections=1, dim=1)
        '''
            if c >= 3:
                x = x[:, 0:3, ...]
                rgb = [0.299, 0.587, 0.114]
                rgb = torch.Tensor(rgb).view(1, 3, 1, 1).expand(b, 3, h, w).cuda()
                x = torch.sum((x * rgb), dim=1, keepdim=True)
        '''
        grad_xs = []
        grad_ys = []
        grads = []
        for i in range(c):
            grad_x = F.conv2d(xs[i], self.weight_x, padding=1)
            grad_xs.append(grad_x)
            grad_y = F.conv2d(xs[i], self.weight_y, padding=1)
            grad_ys.append(grad_y)
            grad = torch.abs(grad_x) * 0.5 + torch.abs(grad_y) * 0.5
            grads.append(grad)

        grad_x = torch.cat(grad_xs, dim=1)
        grad_y = torch.cat(grad_ys, dim=1)
        grad = torch.cat(grads, dim=1)

        return grad, torch.abs(grad_x), torch.abs(grad_y)


def gradient(x):
    gradient_model = Gradient().to(device)
    g = gradient_model(x)
    return g

