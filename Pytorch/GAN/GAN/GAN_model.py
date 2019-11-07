
import torch
import torch.nn as nn

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu, in_channels):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nz(72)*1*1 => 1024*1*1
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 1024*1*1 => 128*7*7
            nn.ConvTranspose2d(1024, 128, 7, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128*7*7 => 64*14,14
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64*14*14 => 1*28*28
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

# Discriminator
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 1*28*28 => 64*14*14
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 64*14*14 => 128*7*7
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # 128*7*7 => 1024*1*1
            nn.Conv2d(128, 1024, 7, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.main_netD = nn.Sequential(
            # 1024 => 1
            nn.Conv2d(1024, 1, 1, 1, 0, bias=False),
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


