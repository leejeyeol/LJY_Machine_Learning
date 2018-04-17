import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, num_in_channels, z_size=200, num_filters=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227

            nn.Linear(1080,1024),

            nn.Conv2d(num_in_channels, num_filters, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.Conv2d(num_filters, 2 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * num_filters, 4 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.Conv2d(4 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.Conv2d(8 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True)
            # state size: (8 x nf) x 6 x 6
        )
        self.z = nn.Conv2d(8 * num_filters, z_size, 6)

        # init weights
        self.weight_init()

    def encode(self, x):
        return self.z(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        return  z

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.z.apply(weight_init)
        self.decoder.apply(weight_init)

class decoder(nn.Module):
    def __init__(self, num_in_channels, z_size=200, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, 8 * num_filters, 6, 2, 0),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * num_filters, 4 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.ConvTranspose2d(4 * num_filters, 2 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 55 x 55
            nn.ConvTranspose2d(2 * num_filters, num_filters, 5, 2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(num_filters, num_in_channels, 5, 2, 1),
            nn.Tanh()
            # state size: (L) x 227 x 227
        )

        # init weights
        self.weight_init()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, z):
        return self.decode(z)

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.z.apply(weight_init)
        self.decoder.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)