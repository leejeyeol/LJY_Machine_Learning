import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import matplotlib.pyplot as plt

import Pytorch.Autoencoder.VGG16_lossnet as vgg
from torchvision import models

# import custom package
import LJY_utils
import LJY_visualize_tools

win_dict = LJY_visualize_tools.win_dict()
line_win_dict = LJY_visualize_tools.win_dict()

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--autoencoderType', default='VAE', help='additional autoencoder type.', choices=['AE', 'VAE', 'AAE'])
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks. '0' is training from scratch.")

parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=100000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=100, help='number of input channel.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)

class encoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64 ,type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 4 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * num_filters, z_size, 3, 1, 0),
        )
        self.fc_mu = nn.Conv2d(z_size, z_size, 1)
        self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            # AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar

    def weight_init(self):
        self.encoder.apply(weight_init)


class decoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 5, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, 5, 1, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, 5, 2, 0),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 0),
            nn.Tanh()
        )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)
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

# save directory make   ================================================================================================
try:
    os.makedirs(options.outf)
except OSError:
    pass

# seed set  ============================================================================================================
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

# cuda set  ============================================================================================================
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

# MNIST call and load   ================================================================================================


'''
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

dataloader = torch.utils.data.DataLoader(
    dset.CIFAR10('../../CIFAR10', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(224),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
'''
# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)

#=======================================================================================================================
# Models
#=======================================================================================================================



# Generator ============================================================================================================
encoder = encoder(num_in_channels=options.nc, z_size=options.nz, num_filters=64, type =options.autoencoderType)
encoder.apply(LJY_utils.weights_init)
print(encoder)

# Discriminator ========================================================================================================
decoder = decoder(num_in_channels=options.nc ,z_size=options.nz, num_filters=64)
decoder.apply(LJY_utils.weights_init)
print(decoder)

loss_net = vgg.VGG16()
print(loss_net)

vggnet = models.vgg16(pretrained=True)
print(vggnet)

#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average=False)
def Perceptual_loss(input, target):
    feature_input = loss_net(input)
    feature_target = loss_net(target)
    loss = 0
    for i in range(4):
        loss = loss + MSE_loss(feature_input[i], feature_target[i])
    return loss/4

def Variational_loss(input, target, mu, logvar):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    return recon_loss, KLD_loss

# setup optimizer   ====================================================================================================
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
#optimizerD = optim.SGD(decoder.parameters(), lr=2e-4)
#optimizerE = optim.SGD(encoder.parameters(),  lr=2e-4)


# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    input = input.cuda()


# make to variables ====================================================================================================
input = Variable(input)

def train():
    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()

    ep = options.pretrainedEpoch
    if ep != 0:
        encoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))

    print("Training Start!")
    for epoch in range(options.iteration):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            # train with real data  ========================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)



            if options.autoencoderType == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err = MSE_loss(x_recon, input.detach())
                err.backward(retain_graph=True)

            elif options.autoencoderType == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_recon, err_kld = Variational_loss(x_recon, input.detach(),mu, logvar )
                err = err_recon + err_kld
                err.backward(retain_graph=True)

            #err_image = (input.detach()-x_recon.detach()).abs()
            #err_perceptual = Perceptual_loss(x_recon, input.detach())
            #err_perceptual.backward(retain_graph=True)


            optimizerE.step()
            optimizerD.step()

            #visualize
            print('[%d/%d][%d/%d] Loss: %.4f'% (epoch, options.iteration, i, len(dataloader),err.data.mean()))
            testImage = torch.cat((input.data[0], x_recon.data[0]), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])
            #line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict, [err.data.mean(),0], ['loss_recon_x','zero'], epoch, i, len(dataloader))
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict, [err_recon.data.mean(),err_kld.data.mean(), 0],
                                                                      ['loss_recon_x','loss kld', 'zero'], epoch, i, len(dataloader))

        # do checkpointing
        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), '%s/encoder_latent_test_z_%d_epoch_%d.pth' % (options.outf, options.nz, epoch))
            torch.save(decoder.state_dict(), '%s/decoder_latent_test_z_%d_epoch_%d.pth' % (options.outf, options.nz, epoch))
def test():
    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=1, shuffle=True, num_workers=options.workers)
    ep = 1000
    if ep != 0:
        encoder.load_state_dict(
            torch.load('%s/encoder_latent_test_z_%d_epoch_%d.pth' % (options.outf, options.nz, ep)))
        decoder.load_state_dict(
            torch.load('%s/decoder_latent_test_z_%d_epoch_%d.pth' % (options.outf, options.nz, ep)))
    z_tensor = None
    for epoch in range(1):
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)


            if options.autoencoderType == 'AE':
                z = encoder(input)
            elif options.autoencoderType == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
            if z_tensor is None:
                z_tensor = z.data

            else:
                z_tensor = torch.cat((z_tensor, z.data), 0)


            # visualize
            print('[%d/%d][%d/%d]' % (epoch, options.iteration, i, len(dataloader)))
    z_hist = torch.var(z_tensor, 0).view(-1).cpu().numpy()
    x=torch.range(1, 1000).numpy()
    plt.plot(x,z_hist)
    plt.show()
    print('end')

if __name__ == "__main__" :
    train()
    #test()
    #tsne()
    #visualize_latent_space_2d()



# Je Yeol. Lee \[T]/
# Jolly Co-operation