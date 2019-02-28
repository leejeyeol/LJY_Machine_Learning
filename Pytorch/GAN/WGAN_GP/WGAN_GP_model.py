import torch.nn as nn
import torch
# Generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

'''
class Generator(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 5, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 5, 1, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 5, 2, 0),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 0),
            nn.Tanh()
        )


    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x



class Discriminator(nn.Module):
    def __init__(self, num_in_channels=1,  num_filters=64):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 4 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * num_filters, 1, 3, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        d = self.discriminator(x)
        return d


'''
'''
#150x150
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 512*4*4 => 512*8*8
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 512*8*8 => 256*18*18
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 256*18*18 => 128*36*36
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 128*36*36 => 64*74*74
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64*74*74 => 3*150*150
            nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

'''
# https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch/blob/master/models_64x64.py
class Generator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class DiscriminatorWGANGP(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

class decoder64x64(nn.Module):
    def __init__(self, num_in_channels=3, z_size=2, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

