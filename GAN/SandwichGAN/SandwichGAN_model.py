import torch
import torch.nn as nn

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main_encoder = nn.Sequential(
            # 6*64*64 => 128*58*58
            nn.Conv2d(in_channels=6, out_channels=128, kernel_size=9, stride=4, padding=1,bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            # 128*58*58 => 64*52*52
            nn.Conv2d(128, 256, 7, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # 64*52*52 => 32*48*48
            nn.Conv2d(256, 512, 5, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            # 32*48*48 => 32*46*46
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.main_netG = nn.Sequential(
            # 32*46*46 => 32*52*52
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 32*52*52 => 64*58*58
            nn.ConvTranspose2d(1024, 512, 4,2,1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 64*58*58 => 128*62*62
            nn.ConvTranspose2d(512, 256, 4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 64*58*58 => 128*62*62
            nn.ConvTranspose2d(256, 128, 4,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128*62*62 => 3*64*64
            nn.ConvTranspose2d(128, 3, 4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main_encoder, input, range(self.ngpu))
            outputG = nn.parallel.data_parallel(self.main_netG, output, range(self.ngpu))
        else:
            output = self.main_encoder(input)
            outputG = self.main_netG(output)
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

