import torch
import torch.nn as nn
class encoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=200, num_filters=64):
        super().__init__()
        self.layer_1 = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_in_channels, num_filters, 5, 4, 3),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = nn.Sequential(
            # state size: (nf) x 113 x 113
            nn.Conv2d(num_filters, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_3 = nn.Sequential(
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * num_filters, z_size, 3, 2, 0)
        )



        # init weights
        self.weight_init()

    def forward(self, x):
        feature_map_1 = self.layer_1(x)
        feature_map_2 = self.layer_2(feature_map_1)
        z = self.layer_3(feature_map_2)

        return z

    def weight_init(self):
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)


class decoder(nn.Module):
    def __init__(self, num_in_channels=1 ,z_size=200, num_filters=64):
        super().__init__()

        self.layer_1 = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, num_filters*2, 3, 1, 0),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = nn.Sequential(
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d( num_filters*2,  num_filters, 4, 3, 1),
            nn.BatchNorm2d( num_filters),
            nn.LeakyReLU(0.2, True),
        )

        self.layer_3 = nn.Sequential(
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(num_filters, num_in_channels, 6, 4, 3),
            nn.Tanh()
        )
            # state size: (L) x 227 x 227


        # init weights
        self.weight_init()

    def forward(self, z):
        feature_map_1 = self.layer_1(z)
        feature_map_2 = self.layer_2(feature_map_1)
        recon_x = self.layer_3(feature_map_2)
        return recon_x

    def weight_init(self):
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)
'''
64x64
class encoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=200, num_filters=64):
        super().__init__()
        self.layer_1 = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_in_channels, num_filters, 5, 4, 3),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = nn.Sequential(
            # state size: (nf) x 113 x 113
            nn.Conv2d(num_filters, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_3 = nn.Sequential(
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * num_filters, 4 * num_filters, 3, 2, 0),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_4 = nn.Sequential(
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(4 * num_filters, z_size, 3, 1, 0),
        )


        # init weights
        self.weight_init()

    def forward(self, x):
        feature_map_1 = self.layer_1(x)
        feature_map_2 = self.layer_2(feature_map_1)
        feature_map_3 = self.layer_3(feature_map_2)
        z = self.layer_4(feature_map_3)

        return z, [feature_map_1, feature_map_2, feature_map_3]

    def weight_init(self):
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)
        self.layer_4.apply(weight_init)


class decoder(nn.Module):
    def __init__(self, num_in_channels=1 ,z_size=200, num_filters=64):
        super().__init__()

        self.layer_1 = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, 8 * num_filters, 3, 1, 0),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = nn.Sequential(
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, 4, 1, 0),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
        )
        self.layer_3 = nn.Sequential(
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * num_filters, 4 * num_filters, 4, 3, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True)
        )

        self.layer_4 = nn.Sequential(
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(4 * num_filters, num_in_channels, 6, 4, 3),
            nn.Tanh()
        )
            # state size: (L) x 227 x 227


        # init weights
        self.weight_init()

    def forward(self, z):
        feature_map_1 = self.layer_1(z)
        feature_map_2 = self.layer_2(feature_map_1)
        feature_map_3 = self.layer_3(feature_map_2)
        recon_x = self.layer_4(feature_map_3)
        return recon_x , [feature_map_1,feature_map_2,feature_map_3]

    def weight_init(self):
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)
        self.layer_4.apply(weight_init)
'''

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)