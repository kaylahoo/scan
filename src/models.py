import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
from .GaussianBlur import GaussianBlur
from .Gradient import gradient
from skimage.feature import canny
from skimage.color import rgb2gray
from .utils import imsave

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + mask(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks, dsts, residuals):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        others = dsts
        outputs_rough, orth_loss1 = self(images, masks, others, stage=0)
        inputs_fine = images * masks.float() + outputs_rough * (1 - masks.float())
        outputs_fine, orth_loss2 = self(inputs_fine, masks, residuals, stage=1)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs_fine.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs_fine
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss_rough
        gen_l1_loss1 = self.l1_loss(outputs_rough, images) * self.config.L1_LOSS_WEIGHT / torch.mean(1-masks)
        # outputs_edge = get_edge(outputs_rough)
        gradient, outputs_gradx, outputs_grady = get_gradient_map(outputs_rough)
        gen_l1_structure = self.l1_loss(gradient, dsts) / torch.mean(1-masks)
        gen_l1_loss1 = gen_l1_loss1 + gen_l1_structure


        # generator l1 loss_fine
        gen_l1_loss2 = self.l1_loss(outputs_fine, images) * self.config.L1_LOSS_WEIGHT / torch.mean(1-masks)
        outputs_residual = get_residual(outputs_fine)
        gen_l1_residual = self.l1_loss(outputs_residual, residuals) / torch.mean(1-masks)
        gen_l1_loss2 = gen_l1_loss2 + gen_l1_residual

        gen_l1_loss = gen_l1_loss1 + gen_l1_loss2

        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs_fine, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs_fine, images)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # orth_loss
        gen_orth_loss = orth_loss1 + orth_loss2
        gen_loss += gen_orth_loss

        logs = [
            ("l_2_d2", dis_loss.item()),
            ("l_2_g2", gen_gan_loss.item()),
            ("l_2_l1", gen_l1_loss.item()),
            ("l_2_per", gen_content_loss.item()),
            ("l_2_sty", gen_style_loss.item()),
            ("l_2_orth", gen_orth_loss.item()),
            ('gen_2_loss', gen_loss.item())
        ]

        return outputs_fine, gen_loss, dis_loss, logs

    def forward(self, images, masks, others, stage):
        images_masked = images * masks.float()
        if stage is 'test':
            images_masked = images * masks.float()
            others = torch.split(others, split_size_or_sections=3, dim=1)
            structure_masked = others[0] * masks.float()
            residual_masked = others[1] * masks.float()
            inputs_rough = torch.cat((images_masked, masks, structure_masked), dim=1)
            outputs, _ = self.generator(inputs_rough, masks, stage=0)
            outputs_merged = images_masked + outputs * (1 - masks.float())
            inputs_fine = torch.cat((outputs_merged, masks, residual_masked), dim=1)
            outputs, _ = self.generator(inputs_fine, masks, stage=1)
            return outputs, _

        if stage is 0:
            images_masked = images * masks.float()  # hole:0
        if stage is 1:
            images_masked = images
        others_masked = others * masks.float()
        inputs = torch.cat((images_masked, masks, others_masked), dim=1)  # input:7
        outputs, orth_loss = self.generator(inputs, masks, stage)
        return outputs, orth_loss

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img


def to_tensor(img):
    img = img / 255.0
    img_t = img.permute(0, 3, 1, 2)
    return img_t


def get_edge(img):
    imgs_ = list(torch.split(postprocess(img), split_size_or_sections=1, dim=0))
    imgs = []
    for i in imgs_:
        i = torch.squeeze(i, dim=0)
        i = i.detach().numpy()
        imgs.append(to_tensor(canny(rgb2gray(i), sigma=2)))
    return torch.cat(imgs, dim=0)


def get_gradient_map(img):
    return gradient(img)


def get_residual(img, low='gaussian'):
    if isinstance(low, str) and low == 'gaussian':
        gaussian_blur = GaussianBlur(9, 10)
        low = gaussian_blur(img)
    high = img - low
    return high
