import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227

            nn.Linear(1011,1024),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(1024,512),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.Dropout(),
            nn.ReLU()
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
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 1011),
            nn.Sigmoid()
            # state size: (L) x 227 x 227
        )

        # init weights
        self.weight_init()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, z):
        return self.decode(z)

    def weight_init(self):

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