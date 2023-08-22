# """
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# @author: Junguang Jiang
# @contact: JiangJunguang1123@outlook.com
# """
import torch.nn as nn
import torch
# from torch.nn import init
# import functools


# class Identity(nn.Module):
#     def forward(self, x):
#         return x


# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.

#     Args:
#         net (torch.nn.Module): network to be initialized
#         init_type (str): the name of an initialization method. Choices includes: ``normal`` |
#             ``xavier`` | ``kaiming`` | ``orthogonal``
#         init_gain (float): scaling factor for normal, xavier and orthogonal.

#     'normal' is used in the original CycleGAN paper. But xavier and kaiming might
#     work better for some applications.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>


# def get_norm_layer(norm_type='instance'):
#     """Return a normalization layer

#     Parameters:
#         norm_type (str) -- the name of the normalization layer: batch | instance | none

#     For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
#     For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
#     """
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         def norm_layer(x): return Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer


# class NLayerDiscriminator(nn.Module):
#     """Construct a PatchGAN discriminator

#     Args:
#         input_nc (int): the number of channels in input images.
#         ndf (int): the number of filters in the last conv layer. Default: 64
#         n_layers (int): the number of conv layers in the discriminator. Default: 3
#         norm_layer (torch.nn.Module): normalization layer. Default: :class:`nn.BatchNorm2d`
#     """

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         return self.model(input)


# class PixelDiscriminator(nn.Module):
#     """Construct a 1x1 PatchGAN discriminator (pixelGAN)

#     Args:
#         input_nc (int): the number of channels in input images.
#         ndf (int): the number of filters in the last conv layer. Default: 64
#         norm_layer (torch.nn.Module): normalization layer. Default: :class:`nn.BatchNorm2d`
#     """

#     def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
#         super(PixelDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         self.net = [
#             nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

#         self.net = nn.Sequential(*self.net)

#     def forward(self, input):
#         return self.net(input)


# def patch(ndf, input_nc=3, norm='batch', n_layers=3, init_type='normal', init_gain=0.02):
#     """
#     PatchGAN classifier described in the original pix2pix paper.
#     It can classify whether 70×70 overlapping patches are real or fake.
#     Such a patch-level discriminator architecture has fewer parameters
#     than a full-image discriminator and can work on arbitrarily-sized images
#     in a fully convolutional fashion.

#     Args:
#         ndf (int): the number of filters in the first conv layer
#         input_nc (int): the number of channels in input images. Default: 3
#         norm (str): the type of normalization layers used in the network. Default: 'batch'
#         n_layers (int): the number of conv layers in the discriminator. Default: 3
#         init_type (str): the name of the initialization method. Choices includes: ``normal`` |
#             ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
#         init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
#     """
#     norm_layer = get_norm_layer(norm_type=norm)
#     net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer)
#     init_weights(net, init_type, init_gain=init_gain)
#     return net


# def pixel(ndf, input_nc=3, norm='batch', init_type='normal', init_gain=0.02):
#     """
#     1x1 PixelGAN discriminator can classify whether a pixel is real or not.
#     It encourages greater color diversity but has no effect on spatial statistics.

#     Args:
#         ndf (int): the number of filters in the first conv layer
#         input_nc (int): the number of channels in input images. Default: 3
#         norm (str): the type of normalization layers used in the network. Default: 'batch'
#         init_type (str): the name of the initialization method. Choices includes: ``normal`` |
#             ``xavier`` | ``kaiming`` | ``orthogonal``. Default: 'normal'
#         init_gain (float): scaling factor for normal, xavier and orthogonal. Default: 0.02
#     """
#     norm_layer = get_norm_layer(norm_type=norm)
#     net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
#     init_weights(net, init_type, init_gain=init_gain)
#     return net


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=7):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=1, stride=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_chs):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_chs, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, stride=2, padding=1)
        )

    def forward(self, img):
        return self.model(img)