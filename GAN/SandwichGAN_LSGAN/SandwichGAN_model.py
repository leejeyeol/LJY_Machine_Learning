import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class _encoder(nn.Module):
    def __init__(self, ngpu):
        super(_encoder, self).__init__()
        self.ngpu = ngpu
        self.encoder_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder_layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder_residual_block = nn.Sequential(
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256))


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self._encoder, input, range(self.ngpu))
        else:
            output_1 = self.encoder_layer_1(input) # 80 64 64 64
            output_2 = self.encoder_layer_2(output_1) # 80 128 32 32
            output_3 = self.encoder_layer_3(output_2) # 80 256 16 16
            output = self.encoder_residual_block(output_3) # 80 256 16 16



        return output, (output_1, output_2)

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu

        self.netG_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5))

        self.netG_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4,2,1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5))

        self.netG_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 7,1,3, bias=False),
            nn.Tanh())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputG = nn.parallel.data_parallel(self.main_netG, input, range(self.ngpu))
        else:
            output_1 = self.netG_layer_1(input) # 80 128 32 32
            output_2 = self.netG_layer_2(output_1) # 80 64 64 64
            outputG = self.netG_layer_3(output_2) # 80 3 64 64

        return outputG, (output_2, output_1)

# Discriminator
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=64, kernel_size=4, stride=2, padding=1,bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),


            nn.Conv2d(1024, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputD = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            outputD = self.main(input)
        return outputD



'''
class _encoder(nn.Module):
    def __init__(self, ngpu):
        super(_encoder, self).__init__()
        self.ngpu = ngpu
        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=4, padding=1,bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 7, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 5, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self._encoder, input, range(self.ngpu))
        else:
            output = self._encoder(input)
        return output

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu

        self.main_netG = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256*2, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4,2,1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4,2,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputG = nn.parallel.data_parallel(self.main_netG, input, range(self.ngpu))
        else:
            outputG = self.main_netG(input)
        return outputG

# Discriminator
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 9*64*64 => 16*32*32
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=4, stride=2, padding=1,bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # 16*32*32 => 32*16*16
            nn.Conv2d(16, 32, 4, 2, 1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # 32*16*16 => 64*8*8
            nn.Conv2d(32, 64, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # 64*8*8 => 128*4*4
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.main_netD = nn.Sequential(
            # 128*4*4 => 1*1*1
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            outputD = nn.parallel.data_parallel(self.main_netD, output, range(self.ngpu))
        else:
            output = self.main(input)
            outputD = self.main_netD(output)
        return outputD

'''