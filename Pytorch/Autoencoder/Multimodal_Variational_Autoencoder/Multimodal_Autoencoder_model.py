import torch
import torch.nn as nn
from torch.autograd import Variable

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (4, 11), 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, (4, 11), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, (4, 11), 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, (4, 11), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (4, 11), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (3, 12), 1, 0),
            nn.LeakyReLU(0.2, True)
        )
        self.fc_mu = nn.Conv2d(512,512, 1)
        self.fc_sig = nn.Conv2d(512,512, 1)
        # init weights
        self.weight_init()

    def encode(self, x):
        x_ = self.encoder(x)
        mu = self.fc_mu(x_)
        logvar = self.fc_sig(x_)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5*logvar)
        eps = Variable(torch.randn(std.size()), requires_grad = False).cuda()
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def weight_init(self):
        self.encoder.apply(weight_init)


class decoder(nn.Module):
    def __init__(self, data_type="rgb"):
        super().__init__()
        self.data_type = data_type
        #2048
        self.decoder_RGB = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, (3, 12), 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(512, 512, (4, 11), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(512, 256, (4, 11), 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, (4, 12), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, (4, 12), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(64, 1, (4, 12), 2, 1),
            nn.Sigmoid()
        )
        self.decoder_d = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, (3, 12), 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(512, 512, (4, 11), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(512, 256, (4, 11), 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, (4, 12), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, (4, 12), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(64, 1, (4, 12), 2, 1),
            nn.Sigmoid()
        )

        # init weights
        self.weight_init()

    def decode(self, z):
        if self.data_type == "rgb":
            return self.decoder_RGB(z)
        elif self.data_type == "d":
            return self.decoder_d(z)

    def forward(self, z):
        return self.decode(z)

    def weight_init(self):
        if self.data_type == "rgb":
            self.decoder_RGB.apply(weight_init)
        elif self.data_type == "d":
            self.decoder_d.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

'''

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #512
        self.encoder = nn.Sequential(
            nn.Linear(1080,1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 24),
            nn.LeakyReLU(0.2, True)
        )
        # init weights
        self.weight_init()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return z

    def weight_init(self):
        self.encoder.apply(weight_init)


class decoder(nn.Module):
    def __init__(self, data_type="rgb"):
        super().__init__()
        self.data_type = data_type
        #2048
        self.decoder_RGB = nn.Sequential(
            nn.Linear(96, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1080),
            nn.Tanh()
        )
        self.decoder_d = nn.Sequential(
            nn.Linear(96, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1080),
        )

        # init weights
        self.weight_init()

    def decode(self, z):
        if self.data_type == "rgb":
            return self.decoder_RGB(z)
        elif self.data_type == "d":
            return self.decoder_d(z)

    def forward(self, z):
        return self.decode(z)

    def weight_init(self):
        if self.data_type == "rgb":
            self.decoder_RGB.apply(weight_init)
        elif self.data_type == "d":
            self.decoder_d.apply(weight_init)
'''