import torch.utils.data as ud
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
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

import math
import glob as glob

from PIL import Image
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from Pytorch.GAN.mixture_gaussian import data_generator

import LJY_utils
import LJY_visualize_tools

plt.style.use('ggplot')

# =======================================================================================================================
# Options
# =======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='MNIST', help='what is dataset? MG : Mixtures of Gaussian',
                    choices=['CelebA', 'MNIST', 'biasedMNIST', 'MNIST_MC', 'MG', 'CIFAR10'])
parser.add_argument('--dataroot',
                    default='/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png',
                    help='path to dataset')
parser.add_argument('--img_size', type=int, default=0, help='0 is default of dataset. 224,112,56,28')
parser.add_argument('--intergrationType', default='intergration', help='additional autoencoder type.',
                    choices=['AEonly', 'GANonly', 'intergration'])
parser.add_argument('--autoencoderType', default='AE', help='additional autoencoder type.',
                    choices=['AE', 'VAE', 'AAE', 'GAN', 'RAE'])
parser.add_argument('--ganType', default='DCGAN', help='additional autoencoder type. "GAN" use DCGAN only',
                    choices=['DCGAN', 'small_D', 'NoiseGAN', 'InfoGAN'])
parser.add_argument('--pretrainedEpoch', type=int, default=0,
                    help="path of Decoder networks. '0' is training from scratch.")
parser.add_argument('--pretrainedModelName', default='CelebA_Test1000vae_recon', help="path of Encoder networks.")
parser.add_argument('--modelOutFolder', default='/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MNIST',
                    help="folder to model checkpoints")
parser.add_argument('--resultOutFolder', default='./results', help="folder to test results")
parser.add_argument('--save_tick', type=int, default=1, help='save tick')
parser.add_argument('--display_type', default='per_epoch', help='displat tick', choices=['per_epoch', 'per_iter'])

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--save', default=True, help='save options. default:False. NOT IMPLEMENTED')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=50000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=2, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
criterion = nn.CrossEntropyLoss()


def Variational_loss(input, target, mu, logvar):
    alpha = 1
    beta = 0.01
    recon_loss = L1_loss(input, target)
    batch_size = logvar.data.shape[0]
    nz = logvar.data.shape[1]
    KLD_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (nz * batch_size)
    # print(input.data[:,1].max())
    # print(input.data[:,1].min())
    return alpha * recon_loss, beta * KLD_loss


def pepper_noise(ins, is_training, prob=0.9):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        return torch.mul(ins, mask)
    return ins


def gaussian_noise(ins, is_training, mean=0, stddev=1, prob=0.1):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev)) / 255
        noise = torch.mul(noise, mask)
        return ins + noise
    return ins


def add_noise(ins):
    noise = (torch.randn(ins.size()) * 0.2).cuda()
    noisy_img = ins.data + noise
    return Variable(noisy_img)



class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def foward(self, x):
        return x * F.sigmoid(x)


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)



class Encoder32x32(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 3, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 3, 2, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, 3, 2, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, z_size, 4, 1, 0),

        )
        self.fc_mu = nn.Conv2d(z_size, z_size, 1)
        self.fc_sig = nn.Conv2d(z_size, z_size, 1)

    def forward(self, x):
        if self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
        else:
            # AE
            z = self.encoder(x)
            return z

class Classifier(nn.Module):
    def __init__(self, z_size=80):
        super().__init__()
        self.type = type
        self.classifier = nn.Sequential(
            nn.Linear(z_size,z_size),
            Swish(),

            nn.Linear(z_size, 2*z_size),
            Swish(),

            nn.Linear(2*z_size, 10),
        )


    def forward(self, x):

            cls = self.classifier(x.view(-1, 1))
            return cls

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal(module.weight.data)


# save directory make   ================================================================================================
try:
    os.makedirs(options.modelOutFolder)
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

# =======================================================================================================================
# Data and Parameters
# =======================================================================================================================

# MNIST call and load   ================================================================================================

ngpu = int(options.ngpu)
nz = int(options.nz)
autoencoder_type = options.autoencoderType
if options.dataset == 'CIFAR10':
    encoder = Encoder32x32(num_in_channels=3, z_size=nz, type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    classifier = Classifier(z_size=nz)
    classifier.apply(LJY_utils.weights_init)
    print(classifier)


# =======================================================================================================================
# Training
# =======================================================================================================================


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
#optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizer = optim.SGD([encoder.parameters(),classifier.parameters()], lr=0.001, momentum=0.9)
#optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

if options.cuda:
    encoder.cuda()
    classifier.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    criterion.cuda()

# training start
def train():

    ep = options.pretrainedEpoch
    if ep != 0:
        encoder.load_state_dict(
            torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))

    if options.dataset == 'CIFAR10':
        dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10(root='../../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                         ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        val_dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10('../../data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                         ])),
            batch_size=options.batchSize, shuffle=False, num_workers=options.workers)


    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()

    print("Training Start!")

    for epoch in range(options.epoch):
        for i, (data, label) in enumerate(dataloader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()
            disc_input = input.clone()

            if autoencoder_type == "VAE":
                optimizer.zero_grad()

                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
            elif autoencoder_type == 'AE' or autoencoder_type == 'AAE':
                optimizer.zero_grad()
                z = encoder(input)

            output = classifier(z)
            err = criterion(output, label)

            err.backward()
            optimizer.step()




            print('[%d/%d][%d/%d]  err: %.4f'
                  % (epoch, options.epoch, i, len(dataloader), err.data.mean()))


        print("Validation Start!")
        for j, (data, label) in enumerate(val_dataloader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()

            if autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
            else:
                z = encoder(input)

            output = classifier(z)
            val_err = criterion(output, label)

        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                  [err.data.mean(),
                                                                   val_err.data.mean(),
                                                                   0],
                                                                  ['train err',
                                                                   'val err',
                                                                   'zero'], 0, epoch, 0)
        # do checkpointing
        if epoch % options.save_tick == 0 or options.save:
            torch.save(encoder.state_dict(), os.path.join(options.modelOutFolder,
                                                          options.pretrainedModelName + "_encoder" + "_%d.pth" % (
                                                                      epoch + ep)))

            print(os.path.join(options.modelOutFolder,
                               options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch + ep)))



def test(modelname, ep):
    if ep != 0:
        encoder.load_state_dict(
            torch.load(os.path.join(options.modelOutFolder, modelname + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(
            torch.load(os.path.join(options.modelOutFolder, modelname + "_decoder" + "_%d.pth" % ep)))
        discriminator.load_state_dict(torch.load(
            os.path.join(options.modelOutFolder, modelname + "_discriminator" + "_%d.pth" % ep)))
        z_discriminator.load_state_dict(torch.load(
            os.path.join(options.modelOutFolder, modelname + "_z_discriminator" + "_%d.pth" % ep)))

    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=100, shuffle=False, num_workers=options.workers)

    vis_x = []
    vis_y = []
    vis_label = []
    print("Testing Start!")
    for j, (data, label) in enumerate(dataloader, 0):
        real_cpu = data
        batch_size = real_cpu.size(0)
        input = Variable(real_cpu).cuda()

        z = encoder(input)
        '''
        mu, logvar = encoder(input)
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
        z = eps.mul(std).add_(mu)
        '''
        zd = z.data.view(batch_size, nz)
        for i in range(batch_size):
            vis_x.append(zd[i][0])
            vis_y.append(zd[i][1])
            vis_label.append(int(label[i]))
        print("[%d/%d]" % (j, len(dataloader)))
    for j in range(int(len(dataloader) / 10)):
        for i in range(batch_size):
            vis_x.append(zd.normal_(0, 1)[i][0])
            vis_y.append(zd.normal_(0, 1)[i][1])
            vis_label.append(int(11))

    fig = plt.figure()

    plt.scatter(vis_x, vis_y, c=vis_label, s=2, cmap='rainbow')
    cb = plt.colorbar()
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(os.path.join('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MNIST_IMG/AAE',
                             "ours_%06d.png" % ep))
    plt.close()
    # plt.show()




if __name__ == "__main__":
    train()
    #test('MNIST_AAEGAN',100)

# Je Yeol. Lee \[T]/
# Jolly Co-operation.tolist()