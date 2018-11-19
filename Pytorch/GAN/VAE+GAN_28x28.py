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

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CIFAR10', help='what is dataset? MG : Mixtures of Gaussian', choices=['CelebA', 'MNIST', 'biasedMNIST', 'MNIST_MC', 'MG','CIFAR10'])
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png', help='path to dataset')
parser.add_argument('--img_size', type=int, default=0, help='0 is default of dataset. 224,112,56,28')
parser.add_argument('--intergrationType', default='intergration', help='additional autoencoder type.', choices=['AEonly', 'GANonly', 'intergration'])
parser.add_argument('--autoencoderType', default='AAE', help='additional autoencoder type.',  choices=['AE', 'VAE', 'AAE', 'GAN', 'RAE'])
parser.add_argument('--ganType',  default='DCGAN', help='additional autoencoder type. "GAN" use DCGAN only', choices=['DCGAN','small_D','NoiseGAN','InfoGAN'])
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks. '0' is training from scratch.")
parser.add_argument('--pretrainedModelName', default='CelebA_Test1000_recon', help="path of Encoder networks.")
parser.add_argument('--modelOutFolder', default='/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MNIST', help="folder to model checkpoints")
parser.add_argument('--resultOutFolder', default='./results', help="folder to test results")
parser.add_argument('--save_tick', type=int, default=1, help='save tick')
parser.add_argument('--display_type', default='per_epoch', help='displat tick',choices=['per_epoch', 'per_iter'])

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--save', default=True, help='save options. default:False. NOT IMPLEMENTED')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=50000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=28, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=64, help='number of input channel.')
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
def Variational_loss(input, target, mu, logvar):
    alpha = 1
    beta = 0.01
    recon_loss = L1_loss(input, target)
    batch_size = logvar.data.shape[0]
    nz = logvar.data.shape[1]
    KLD_loss = (-0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp()))/(nz*batch_size)
    #print(input.data[:,1].max())
    #print(input.data[:,1].min())
    return alpha * recon_loss, beta * KLD_loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    # for siamese network
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
def MGplot_seaborn(MGdset, points, epoch, iteration, total_iter):
    idx = epoch * total_iter + iteration
    points = np.asarray(points.data)
    bg_color = sns.color_palette('Greens', n_colors=256)[0]
    ax = sns.kdeplot(points[:, 0], points[:, 1], shade=True, cmap='Greens', n_levels=20, clip=[[-5, 5]] * 2)
    ax.set_facecolor(bg_color)

    kde = ax.get_figure()
    kde.savefig(os.path.join('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG_ours_size_average_false',"ours_%06d.png" % idx))
    kde.close()

def MGplot(MGdset, points, epoch, iteration, total_iter,seaborn=False):
    if seaborn :
        MGplot_seaborn(MGdset, points, epoch, iteration, total_iter)
    idx = epoch * total_iter + iteration
    plt.figure(figsize=(5,5))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(MGdset.centers[:, 0], MGdset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(os.path.join('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG_ours_size_average_false',"ours_%06d.png" % idx))
    plt.close()

class HMDB51_Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()
        cur_file_paths = []
        HMDB_action_folders = sorted(glob.glob(self.base_path + '/*'))
        for HMDB_actions in HMDB_action_folders:
            HMDB_action = sorted(glob.glob(HMDB_actions + '/*'))
            for clips in HMDB_action:
                clip = sorted(glob.glob(clips + '/*'))
                cur_file_paths = cur_file_paths + clip

        print("data loading complete!")
        self.file_paths = cur_file_paths

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, 1
class biased_MNIST(dset.MNIST):
    def __init__(self, root,  train, download, transform):
        super().__init__(root=root,  train=train, download=download, transform=transform)
        idx_per_label = [[] for _ in range(10)]
        if train :
            label_list = list(self.train_labels)

            for i in range(len(label_list)):
                idx_per_label[label_list[i]].append(i)
            random.seed(1)
            # 0 : 5923 => 5923
            # 1 : 6742 => 2000
            # 2 : 5958 => 2000
            # 3 : 6131 => 2000
            # 4 : 5842 => 2000
            # 5 : 5421 => 500
            # 6 : 5918 => 500
            # 7 : 6265 => 500
            # 8 : 5851 => 500
            # 9 : 5949 => 500

            # container
            unbiased_train_data = self.train_data[0].view(1, 28, 28)
            unbiased_labels = [self.train_labels[0]]

            # sampling data
            for idx in random.sample(idx_per_label[0], 500):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[1], 2000):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[2], 2000):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[3], 2000):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[4], 1000):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[5], 5000):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[6], 400):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[7], 300):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[8], 200):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])
            for idx in random.sample(idx_per_label[9], 100):
                torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
                unbiased_labels.append(self.train_labels[idx])

            self.train_labels = torch.LongTensor(unbiased_labels)
            self.train_data = unbiased_train_data

class MG_Dataloader(torch.utils.data.Dataset):
    def __init__(self, epoch, MG):
        super().__init__()
        self.len = epoch
        self.MG = MG
    def __len__(self):
        return self.len
    def __getitem__(self, item):
        d_real_data = torch.from_numpy(self.MG.sample(1))
        return d_real_data.view(2), 1

class custom_Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

        cur_file_paths = glob.glob(self.base_path + '/*.*')
        cur_file_paths.sort()
        self.file_paths = cur_file_paths

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, 1

def pepper_noise(ins, is_training, prob = 0.9):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        return torch.mul(ins, mask)
    return ins

def gaussian_noise(ins, is_training, mean=0, stddev=1,prob = 0.1):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))/255
        noise = torch.mul(noise, mask)
        return ins + noise
    return ins

def add_noise(ins):
    noise = (torch.randn(ins.size()) * 0.2).cuda()
    noisy_img = ins.data + noise
    return Variable(noisy_img)


def swish(x):
    return x * F.sigmoid(x)


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)

def infogan_noise_sampler(con_c, noise):
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, con_c], 1).view(-1, 74, 1, 1)
    return z
class encoder_freesize(nn.Module):
    def __init__(self,  img_size = 224, num_in_channels=1, z_size=2, num_filters=64 ,type='AE'):
        super().__init__()
        self.type = type
        self.sup_size = [224,112,56,28]
        if img_size in self.sup_size:
            if img_size == self.sup_size[0]:
                self.encoder = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, z_size, 2, 2, 0, bias=False)
                )
            if img_size == self.sup_size[1]:
                self.encoder = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, z_size, 2, 2, 0, bias=False)
                )
            if img_size == self.sup_size[2]:
                self.encoder = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, z_size, 2, 2, 0, bias=False)
                )
            if img_size == self.sup_size[3]:
                self.encoder = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, z_size, 2, 2, 0, bias=False)
                )
        else :
            print('img_size %d is not in sup_size'%img_size)
            return None
        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(z_size, z_size, 1)
            self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            #AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
        else :
            print("autoencoder_type is %s, it is unknown." % self.type)


    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder_freesize(nn.Module):
    def __init__(self, img_size = 224, num_in_channels=3, z_size=2, num_filters=64):
        super().__init__()
        self.sup_size = [224, 112, 56, 28]
        if img_size in self.sup_size:
            if img_size == self.sup_size[0]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(z_size, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters*2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters*2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters*2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters*2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(num_filters, num_in_channels, 2, 2, 1, bias=False),
                    nn.Tanh()
                )
            if img_size == self.sup_size[1]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(z_size, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(num_filters, num_in_channels, 2, 2, 1, bias=False),
                    nn.Tanh()
                )
            if img_size == self.sup_size[2]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(z_size, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(num_filters * 2, num_filters, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(num_filters, num_in_channels, 2, 2, 1, bias=False),
                    nn.Tanh()
                )
            if img_size == self.sup_size[3]:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(z_size, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(num_filters * 2, num_in_channels, 2, 2, 1, bias=False),
                    nn.Tanh()
                )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

class discriminator_freesize(nn.Module):
    def __init__(self, img_size =224, num_in_channels=1,  num_filters=64):
        super().__init__()
        self.sup_size = [224, 112, 56, 28]
        if img_size in self.sup_size:
            if img_size == self.sup_size[0]:
                self.discriminator = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, 1, 2, 2, 0, bias=False),
                    nn.Sigmoid()
                )
            if img_size == self.sup_size[1]:
                self.discriminator = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, 1, 2, 2, 0, bias=False),
                    nn.Sigmoid()
                )
            if img_size == self.sup_size[2]:
                self.discriminator = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, num_filters * 8, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 8, 1, 2, 2, 0, bias=False),
                    nn.Sigmoid()
                )
            if img_size == self.sup_size[3]:
                self.discriminator = nn.Sequential(
                    nn.Conv2d(num_in_channels, num_filters, 3, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters, num_filters * 2, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 2, num_filters * 4, 3, 2, 0, bias=False),
                    nn.BatchNorm2d(num_filters * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_filters * 4, 1, 2, 2, 0, bias=False),
                    nn.Sigmoid()
                )
        # init weights
        self.weight_init()

    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1).squeeze(1)
    def weight_init(self):
        self.discriminator.apply(weight_init)

class encoder224x224(nn.Module):
    '''encoder'''

    def __init__(self, nz, nc, type = 'VAE', large=False):
        super(encoder224x224, self).__init__()
        self.type = type
        self.conv1 = nn.Conv2d(nc, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        if large:
            self.conv6 = nn.Conv2d(512, 512, 15, stride=1, padding=0, bias=False)
        else:
            self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(512, nz, 1, stride=1, padding=0, bias=False)

        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(nz, nz, 1)
            self.fc_sig = nn.Conv2d(nz, nz, 1)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)  # 64,112,112 (if input is 224x224)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)  # 128,56,56

        h = self.conv3(h)  # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h)  # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h)  # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h)  # 512,1,1

        h = self.conv7(h)
        h = F.sigmoid(h)

        if self.type == 'AE' or self.type == 'AAE':
            return h
        elif self.type == 'VAE':
            # VAE
            mu = self.fc_mu(h)
            logvar = self.fc_sig(h)
            return mu, logvar

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class decoder224x224(nn.Module):
    '''Generator'''
    def __init__(self, nz, nc):
        super(decoder224x224, self).__init__()

        self.deconv4 = nn.ConvTranspose2d(nz, 512, 3, stride=2, padding=0,  bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.deconv6 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0,  bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(64, nc, 2, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.deconv4(h)
        h = self.bn4(h)
        h = self.relu4(h)  # 512,3,3

        h = self.deconv5(h)
        h = self.bn5(h)
        h = self.relu5(h)  # 512,7,7

        h = self.deconv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,14,14

        h = self.deconv7(h)
        h = self.bn7(h)
        h = self.relu7(h) # 256,28,28

        h = self.deconv8(h)
        h = self.bn8(h)
        h = self.relu8(h) # 128,56,56

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h) # 64,112,112

        h = self.deconv10(h)
        h = F.tanh(h) # 3,224,224

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class discriminator224x224(nn.Module):
    '''Discriminator'''
    def __init__(self, nc, large=False):
        super(discriminator224x224, self).__init__()

        self.conv1 = nn.Conv2d(nc, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        if large:
            self.conv6 = nn.Conv2d(512, 512, 15, stride=1, padding=0, bias=False)
        else:
            self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h) # 64,112,112 (if input is 224x224)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h) # 128,56,56

        h = self.conv3(h) # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h) # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h) # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,1,1

        h = self.conv7(h)
        h = F.sigmoid(h)

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class encoder64x64(nn.Module):
    def __init__(self,  num_in_channels=1, z_size=2, num_filters=64 ,type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, z_size, 4, 2, 1, bias=False),
        )
        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(z_size, z_size, 1)
            self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            #AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
        else :
            print("autoencoder_type is %s, it is unknown."%self.type)


    def weight_init(self):
        self.encoder.apply(weight_init)

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

        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

class discriminator64x64(nn.Module):
    def __init__(self, num_in_channels=1,  num_filters=64):
        super(discriminator64x64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_in_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # init weights
        self.weight_init()

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    def weight_init(self):
        self.main.apply(weight_init)

class small_discriminator(nn.Module):
    def __init__(self, nz=1):
        super(small_discriminator, self).__init__()
        self.nz= nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nz, 1),
            nn.Sigmoid()
        )
        # init weights
        self.weight_init()

    def forward(self, input):
        output = self.main(input.view(-1, self.nz))
        return output.view(-1, 1).squeeze(1)

    def weight_init(self):
        self.main.apply(weight_init)

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
        # init weights
        self.weight_init()

    def forward(self, x):
        d = self.discriminator(x)
        return d

    def weight_init(self):
        self.discriminator.apply(weight_init)

class siamese_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

        self
        # init weights
        self.weight_init()
    def forward_once(self, x):
        output = x.view(x.size()[0], -1)
        output = self.fc1(output)
        return output
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def weight_init(self):
        self.fc1.apply(weight_init)

class z_discriminator(nn.Module):
    def __init__(self, N=1000, z_dim=120):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, N),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, 1),
            nn.Sigmoid()
        )
        # init weights
        self.weight_init()
    def forward(self, z):
        cls = self.discriminator(z)
        return cls
    def weight_init(self):
        self.discriminator.apply(weight_init)

class encoder_MC(nn.Module):
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

    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder_MC(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64):
        super().__init__()

        self.decoder_fc = nn.Sequential(
            nn.Linear(z_size, 1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.LeakyReLU(0.2, True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128,128,5,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128,1,5,bias=False)
        )
        # init weights
        self.weight_init()

    def forward(self, z):
        z = self.decoder_fc(z)
        recon_x = self.decoder_conv(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

class discriminator_MC(nn.Module):
    def __init__(self, num_in_channels=1,  num_filters=64):
        super().__init__()
        self.discriminator_conv = nn.Sequential(
            nn.Conv2d(num_in_channels, 11, 2),
            nn.BatchNorm2d(11),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(11, 75, 2),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(0.2, True))
        self.discriminator_fc = nn.Sequential(
            nn.Linear(75, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        # init weights
        self.weight_init()

    def forward(self, x):
        d = self.discriminator(x)
        return d

    def weight_init(self):
        self.discriminator.apply(weight_init)

class MG_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MG_decoder, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.ReLU()
        #self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_fn = nn.Tanh()


    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1])
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))

        return self.map3(x)

class MG_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, type):
        super(MG_encoder, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_fn = F.relu
        #self.final_activation_fn = nn.ReLU()
        self.fc_mu = nn.Linear(output_size, output_size)
        self.fc_sig = nn.Linear(output_size, output_size)
        self.type = type

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            # AE
            x = self.activation_fn(self.map1(x))
            x = self.activation_fn(self.map2(x))
            #z = self.final_activation_fn(self.map3(x))
            z =self.map3(x)

            return z
        elif self.type == 'VAE':
            # VAE
            x = self.activation_fn(self.map1(x))
            x = self.activation_fn(self.map2(x))

            #z_ = self.final_activation_fn(self.map3(x))
            z_ = self.activation_fn(self.map3(x))
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar

class MG_discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MG_discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_fn = F.relu
        self.final_activation_fn = F.sigmoid

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))

        return self.final_activation_fn(self.map3(x))


class Info_FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(Info_FrontEnd, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Info_D(nn.Module):

    def __init__(self):
        super(Info_D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Info_Q(nn.Module):

    def __init__(self):
        super(Info_Q, self).__init__()

        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        y = self.conv(x)

        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return mu, var


class Info_G(nn.Module):

    def __init__(self):
        super(Info_G, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output


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

#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

# MNIST call and load   ================================================================================================

ngpu = int(options.ngpu)
nz = int(options.nz)
autoencoder_type = options.autoencoderType
if options.dataset == 'MNIST' or options.dataset == 'biasedMNIST':
    encoder = encoder(num_in_channels=1, z_size=nz, num_filters=64,type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    decoder = decoder(num_in_channels=1, z_size=nz, num_filters=64)
    decoder.apply(LJY_utils.weights_init)
    print(decoder)

    if autoencoder_type == 'RAE':
        Recon_discriminator = Discriminator(num_in_channels=1, num_filters=64)
        Recon_discriminator.apply(LJY_utils.weights_init)
        print(Recon_discriminator)
        Recon_discriminator.cuda()

    if options.ganType == 'small_D':
        discriminator = small_discriminator(nz=nz)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)
    elif options.ganType == 'DCGAN':
        discriminator = Discriminator(num_in_channels=1, num_filters=64)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)
    elif options.ganType == 'NoiseGAN':
        discriminator = Discriminator(num_in_channels=1, num_filters=64)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)

        discriminator_2 = Discriminator(num_in_channels=1, num_filters=64)
        discriminator_2.apply(LJY_utils.weights_init)
        print(discriminator_2)
        discriminator_2.cuda()

elif options.dataset == 'CelebA':
    if options.img_size == 0:
        encoder = encoder64x64(num_in_channels=3, z_size=nz, num_filters=64,type=autoencoder_type)
        encoder.apply(LJY_utils.weights_init)
        print(encoder)

        decoder = decoder64x64(num_in_channels=3, z_size=nz, num_filters=64)
        decoder.apply(LJY_utils.weights_init)
        print(decoder)

        discriminator = discriminator64x64(num_in_channels=3, num_filters=64)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)
        if options.ganType == 'NoiseGAN':
            discriminator = discriminator64x64(num_in_channels=3, num_filters=64)
            discriminator.apply(LJY_utils.weights_init)
            print(discriminator)

            discriminator_2 = discriminator64x64(num_in_channels=3, num_filters=64)
            discriminator_2.apply(LJY_utils.weights_init)
            print(discriminator_2)
            discriminator_2.cuda()
    else:
        encoder = encoder_freesize(img_size=options.img_size, num_in_channels=3, z_size=nz, num_filters=64, type=autoencoder_type)
        encoder.apply(LJY_utils.weights_init)
        print(encoder)

        decoder = decoder_freesize(img_size=options.img_size,num_in_channels=3, z_size=nz, num_filters=64)
        decoder.apply(LJY_utils.weights_init)
        print(decoder)

        discriminator = discriminator_freesize(img_size=options.img_size, num_in_channels=3, num_filters=64)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)


elif options.dataset == 'HMDB51':
    encoder = encoder64x64(num_in_channels=1, z_size=nz, num_filters=64,type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    decoder = decoder64x64(num_in_channels=1, z_size=nz, num_filters=64)
    decoder.apply(LJY_utils.weights_init)
    print(decoder)

    discriminator = discriminator64x64(num_in_channels=1, num_filters=64)
    discriminator.apply(LJY_utils.weights_init)
    print(discriminator)
elif options.dataset == 'HMDB51_224':
    encoder = encoder224x224(options.nz, options.nc)
    print(encoder)

    decoder = decoder224x224(options.nz,  options.nc)
    print(decoder)

    discriminator = discriminator224x224(1)
    print(discriminator)

elif options.dataset == 'MG':
    encoder = MG_encoder(input_size=2, hidden_size=128, output_size=nz, type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    decoder = MG_decoder(input_size=nz, hidden_size=128, output_size=2)
    decoder.apply(LJY_utils.weights_init)
    print(decoder)

    discriminator = MG_discriminator(input_size=2, hidden_size=128, output_size=1)
    discriminator.apply(LJY_utils.weights_init)
    print(discriminator)
z_discriminator = z_discriminator(N=nz*4, z_dim=nz)
z_discriminator.apply(LJY_utils.weights_init)
print(z_discriminator)

#=======================================================================================================================
# Training
#=======================================================================================================================



# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)

#optimizerD = optim.RMSprop(decoder.parameters(), lr=2e-4)
#optimizerE = optim.RMSprop(encoder.parameters(), lr=2e-4)
#optimizerDiscriminator = optim.RMSprop(discriminator.parameters(), lr=2e-4)

#optimizerD = optim.SGD(decoder.parameters(), lr=2e-4)
#optimizerE = optim.SGD(encoder.parameters(), lr=2e-4)
#optimizerDiscriminator = optim.SGD(discriminator.parameters(), lr=2e-3)
if autoencoder_type=='AAE' or autoencoder_type=='RAE':
    optimizer_z_Discriminator = optim.Adam(z_discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)
if autoencoder_type=='RAE':
    optimizer_recon_Discriminator = optim.Adam(Recon_discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)

if options.ganType == 'NoiseGAN':
    optimizer_discriminator_2 = optim.Adam(discriminator_2.parameters(), betas=(0.5, 0.999), lr=2e-3)



if options.cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    z_discriminator.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()


# training start
def train():
    visualize_latent = False
    recon_learn = True
    cycle_learn = False
    recon_weight = 1000.0
    encoder_weight = 1.0
    decoder_weight = 1.0
    validation_path = os.path.join(os.path.dirname(options.modelOutFolder), '%s_%s_%s' % (options.dataset,options.intergrationType, options.autoencoderType))
    validation_path = LJY_utils.make_dir(validation_path, allow_duplication=True)
    save_path = os.path.join(options.modelOutFolder)
    save_path = LJY_utils.make_dir(save_path)
    ep = options.pretrainedEpoch
    if ep != 0:
        encoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
        discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))
        z_discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % ep)))
    if options.dataset == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            dset.MNIST(root='../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        val_dataloader = torch.utils.data.DataLoader(
            dset.MNIST('../../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=100, shuffle=False, num_workers=options.workers)

    elif options.dataset == 'biasedMNIST':
        dataloader = torch.utils.data.DataLoader(
        biased_MNIST(root='../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

        val_dataloader = torch.utils.data.DataLoader(
            biased_MNIST('../../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=100, shuffle=False, num_workers=options.workers)
    elif options.dataset == 'CelebA':
        if options.img_size == 0:
            celebA_imgsize = 64
        else:
            celebA_imgsize = options.img_size

        dataloader = torch.utils.data.DataLoader(
            custom_Dataloader(path=options.dataroot,
                              transform=transforms.Compose([
                                  transforms.CenterCrop(150),
                                  transforms.Scale((celebA_imgsize, celebA_imgsize)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    elif options.dataset == 'HMDB51':
        dataloader = torch.utils.data.DataLoader(
            HMDB51_Dataloader(path=options.dataroot,
                              transform=transforms.Compose([
                                  transforms.Scale((64,64)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ])), batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
    elif options.dataset == 'HMDB51_224':
        dataloader = torch.utils.data.DataLoader(
            HMDB51_Dataloader(path=options.dataroot,
                              transform=transforms.Compose([
                                  transforms.Scale((224,224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ])), batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
    elif options.dataset == 'MG':
        MGdset=data_generator()
        #MGdset.random_distribution()
        MGdset.uniform_distribution()
        dataloader = torch.utils.data.DataLoader(MG_Dataloader(1000, MGdset),batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    grad_line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")



    for epoch in range(options.epoch):
        for i, (data, _) in enumerate(dataloader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()
            disc_input =input.clone()

            real_label = Variable(torch.FloatTensor(batch_size).cuda())
            real_label.data.fill_(1)
            fake_label = Variable(torch.FloatTensor(batch_size).cuda())
            fake_label.data.fill_(0)
            noise_regularizer = Variable(torch.FloatTensor(real_cpu.shape)).cuda()
            noise_regularizer.data.fill_(1)

            # autoencoder part
            if options.intergrationType != 'GANonly':
                if autoencoder_type == "VAE":
                    optimizerE.zero_grad()
                    optimizerD.zero_grad()
                    mu, logvar = encoder(input)
                    std = torch.exp(0.5 * logvar)
                    eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                    z = eps.mul(std).add_(mu)
                    x_recon = decoder(z)
                    err_recon, err_KLD = Variational_loss(x_recon, input.detach(), mu, logvar)
                    err = recon_weight * (err_recon + err_KLD)
                    err.backward(retain_graph=True)
                    generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                    optimizerE.step()
                    optimizerD.step()

                    if cycle_learn:
                        optimizerE.zero_grad()
                        optimizerD.zero_grad()
                        noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
                        noise.data.normal_(0, 1)
                        x = decoder(noise)
                        mu, logvar = encoder(x)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z_recon = eps.mul(std).add_(mu)
                        #x_recon = decoder(z_recon)
                        #err = L1_loss(x_recon, x.detach())
                        err1 = L1_loss(z_recon, noise.detach())
                        err = recon_weight * (err1)
                        err.backward(retain_graph=True)
                        generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                        optimizerE.step()
                        optimizerD.step()

                elif autoencoder_type == 'AE' or autoencoder_type=='AAE':
                    optimizerE.zero_grad()
                    optimizerD.zero_grad()
                    z = encoder(input)
                    x_recon = decoder(z)

                    #z_recon = encoder(x_recon)
                    #err = L1_loss(z_recon, z.detach())
                    err1 = L1_loss(x_recon, input.detach())
                    err = recon_weight * (err1)
                    err.backward(retain_graph=True)
                    generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                    optimizerE.step()
                    optimizerD.step()
                    err_recon = err

                    if cycle_learn:
                        optimizerE.zero_grad()
                        optimizerD.zero_grad()
                        noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
                        noise.data.normal_(0, 1)
                        x = decoder(noise)
                        z_recon = encoder(x)

                        #x_recon = decoder(z_recon)
                        #err = L1_loss(x_recon, x.detach())
                        err1 = L1_loss(z_recon, noise.detach())
                        err = recon_weight * (err1)
                        err.backward(retain_graph=True)
                        generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                        optimizerE.step()
                        optimizerD.step()

                elif autoencoder_type == 'RAE':
                    original_input = input.detach()
                    #noised_input = add_noise(input.detach())
                    noised_input = input.detach()
                    #noised_input = gaussian_noise(input, is_training=True)
                    z = encoder(noised_input.detach())
                    x_recon = decoder(z)
                    for _ in range(5):
                        real_dfimg = noised_input - original_input.detach()
                        fake_dfimg = x_recon - original_input.detach()
                        d_recon_real = Recon_discriminator(real_dfimg)
                        d_recon_fake = Recon_discriminator(fake_dfimg)
                        d_recon_err = BCE_loss(d_recon_fake.view(-1), fake_label) + BCE_loss(d_recon_real.view(-1), real_label)
                        optimizer_recon_Discriminator.zero_grad()
                        d_recon_err.backward(retain_graph=True)
                        optimizer_recon_Discriminator.step()

                    z = encoder(noised_input.detach())
                    x_recon = decoder(z)
                    fake_dfimg = x_recon - original_input.detach()
                    d_rcon = Recon_discriminator(fake_dfimg)
                    err = BCE_loss(d_rcon.view(-1), real_label)

                    optimizerE.zero_grad()
                    optimizerD.zero_grad()
                    err = recon_weight * err
                    err.backward(retain_graph=True)
                    generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                    optimizerE.step()
                    optimizerD.step()

                if autoencoder_type == 'AAE' or autoencoder_type=='RAE':
                    # adversarial autoencoder, latent adversarial loss
                    for _ in range(5):
                        z = encoder(input.detach())
                        z_d_fake = z_discriminator(z.view(z.shape[0], z.shape[1]))  # generated by encoder

                        noise_z = Variable(z.data.view(z.data.shape[0], z.data.shape[1]).normal_(0, 1))
                        z_d_real = z_discriminator(noise_z.view(z.shape[0], z.shape[1]))  # real data (Gaussian Noise)

                        z_err_D  = BCE_loss(z_d_fake.view(-1), fake_label) + BCE_loss(z_d_real.view(-1), real_label)
                        z_err_D = z_err_D
                        optimizer_z_Discriminator.zero_grad()
                        z_err_D.backward(retain_graph=True)
                        optimizer_z_Discriminator.step()
                    optimizerE.zero_grad()
                    z = encoder(input.detach())
                    z_d_fake_2 = z_discriminator(z.view(z.shape[0], z.shape[1]))  # generated by encoder
                    z_err_G = BCE_loss(z_d_fake_2.view(-1), real_label)
                    z_err_G = encoder_weight * z_err_G
                    z_err_G.backward(retain_graph=True)

                    generator_grad_AE += LJY_utils.torch_model_gradient(decoder.parameters())
                    optimizerE.step()
                    print('AAE         z_real : %.4f   z_fake : %.4f    z_fake_2 : %.4f'    % (z_d_real.view(-1).data.mean(), z_d_fake.view(-1).data.mean(),
                                                                                   z_d_fake_2.view(-1).data.mean()))
            print('[%d/%d][%d/%d] recon_error : %.4f'
                  % (epoch, options.epoch, i, len(dataloader), err.mean()))



            # adversarial training part  =======================================================================================
            # adversarial training, discriminator part
            if options.intergrationType != 'AEonly': # GAN
                optimizerDiscriminator.zero_grad()

                if options.ganType  == 'NoiseGAN':
                    d_real = discriminator_2(noise_regularizer)
                    z = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    z.data.normal_(0, 1)
                    noise = decoder(z.view(batch_size, nz, 1, 1))
                    d_fake = discriminator_2(noise)
                    err_discriminator_real = BCE_loss(d_real.view(batch_size), real_label.view(batch_size))
                    err_discriminator_fake = BCE_loss(d_fake.view(batch_size), fake_label.view(batch_size))
                    err_discriminator_origin = err_discriminator_real + err_discriminator_fake
                    err_discriminator = err_discriminator_origin
                    err_discriminator.backward(retain_graph=True)
                    optimizer_discriminator_2.step()

                    z = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    z.data.normal_(0, 1)
                    noise = decoder(z.view(batch_size, nz, 1, 1))
                    d_fake_2 = discriminator_2(noise)
                    err_generator = BCE_loss(d_fake_2.view(batch_size), fake_label.view(batch_size))
                    err_generator = 0.1 * err_generator
                    optimizerD.zero_grad()
                    err_generator.backward(retain_graph=True)
                    optimizerD.step()
                if options.ganType == 'small_D':
                    if autoencoder_type == 'VAE':
                        mu, logvar = encoder(disc_input)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        z = encoder(disc_input)
                    d_real = discriminator(z)
                    noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    noise.data.normal_(0, 1)
                    generated_fake = decoder(noise.view(batch_size, nz, 1, 1))

                    if autoencoder_type == 'VAE':
                        mu, logvar = encoder(generated_fake)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        z = encoder(generated_fake)
                    d_fake = discriminator(z)

                elif options.ganType == 'NoiseGAN':
                    d_real = discriminator(input)
                    z = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    z.data.normal_(0, 1)
                    noise = decoder(z.view(batch_size, nz, 1, 1))
                    generated_fake = input.detach() * noise
                    d_fake = discriminator(generated_fake)
                else:
                    d_real = discriminator(input)
                    noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    noise.data.normal_(0, 1)
                    generated_fake = decoder(noise.view(batch_size, nz, 1, 1))
                    d_fake = discriminator(generated_fake)


                balance_coef = torch.cat((d_fake, d_real), 0).mean()
                err_discriminator_real = BCE_loss(d_real.view(batch_size), real_label.view(batch_size))
                err_discriminator_fake = BCE_loss(d_fake.view(batch_size), fake_label.view(batch_size))
                if recon_learn:
                    if autoencoder_type == 'VAE':
                        mu, logvar = encoder(disc_input)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        z = encoder(input)
                    x_recon = decoder(z)
                    d_auto = discriminator(x_recon)
                    err_discriminator_auto = BCE_loss(d_auto.view(batch_size), fake_label.view(batch_size))
                    err_discriminator_origin = err_discriminator_real + err_discriminator_fake + err_discriminator_auto
                else:
                    err_discriminator_origin = err_discriminator_real + err_discriminator_fake
                err_discriminator = decoder_weight * err_discriminator_origin
                err_discriminator.backward(retain_graph=True)

                discriminator_grad = LJY_utils.torch_model_gradient(discriminator.parameters())
                optimizerDiscriminator.step()

                # adversarial training, generator part
                noise = Variable(torch.FloatTensor(batch_size, nz ,1 ,1)).cuda()
                noise.data.normal_(0, 1)
                if options.ganType == 'small_D':
                    if autoencoder_type == 'VAE':
                        generated_fake = decoder(noise)
                        mu, logvar = encoder(generated_fake)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        generated_fake = decoder(noise)
                        z = encoder(generated_fake)
                    d_fake_2 = discriminator(z)
                elif options.ganType == 'NoiseGAN':

                    noise = decoder(z.view(batch_size, nz, 1, 1))
                    generated_fake = input.detach() * noise
                    d_fake_2 = discriminator(generated_fake)
                else:
                    generated_fake = decoder(noise)
                    d_fake_2 = discriminator(generated_fake)
                if recon_learn:
                    err_generator = BCE_loss(d_fake_2.view(batch_size), real_label.view(batch_size))
                    err_generator = decoder_weight * err_generator
                    optimizerD.zero_grad()
                    err_generator.backward(retain_graph=True)
                    #optimizerE.zero_grad()
                    generator_grad = LJY_utils.torch_model_gradient(decoder.parameters())

                    if autoencoder_type == 'VAE':
                        mu, logvar = encoder(disc_input)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        z = encoder(input)
                    x = decoder(z)
                    d_auto_fake = discriminator(x)
                    err_auto = BCE_loss(d_auto_fake.view(batch_size), real_label.view(batch_size))
                    err_auto.backward()
                    optimizerD.step()
                    #optimizerE.step()

                else:
                    err_generator = BCE_loss(d_fake_2.view(batch_size), real_label.view(batch_size))
                    err_generator = decoder_weight * err_generator
                    optimizerD.zero_grad()
                    err_generator.backward(retain_graph=True)
                    generator_grad = LJY_utils.torch_model_gradient(decoder.parameters())
                    optimizerD.step()

                 #visualize
                print('[%d/%d][%d/%d] d_real: %.4f d_fake: %.4f Balance : %.2f'
                      % (epoch, options.epoch, i, len(dataloader), d_real.data.mean(), d_fake_2.data.mean(), balance_coef.data.mean()))
                #print(float(noise.data.view(noise.shape[0], -1).var(1).mean()))
                #print(float(noise.data.view(noise.shape[0], -1).mean(1).mean()))

            else:
                print('[%d/%d][%d/%d] recon_Loss: AE  err: %.4f'
                      % (epoch, options.epoch, i, len(dataloader), err.data.mean()))

            if options.display:
                if options.dataset != 'MG':
                    if options.intergrationType == 'GANonly':
                        testImage = torch.cat((unorm(input.data[0]), unorm(generated_fake.data[0])), 2)
                    elif options.intergrationType == 'AEonly':
                        testImage = torch.cat((unorm(input.data[0]), unorm(x_recon.data[0])), 2)
                    else:
                        testImage = torch.cat((unorm(input.data[0]), unorm(x_recon.data[0]), unorm(generated_fake.data[0])), 2)
                    win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            if options.display_type == 'per_iter':
                if options.dataset == 'MG':
                    noise = Variable(torch.FloatTensor(1000, nz)).cuda()
                    noise.data.normal_(0, 1)
                    generated_fake = decoder(noise)
                    MGplot(MGdset, generated_fake, epoch, i, len(dataloader))
                if options.intergrationType != 'AEonly':
                    if autoencoder_type == 'VAE':
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [err_discriminator_real.data.mean(),
                                                                                   err_discriminator_fake.data.mean(),
                                                                                   err_generator.data.mean(),
                                                                                   err_recon.data.mean(),
                                                                                   err_KLD.data.mean(),
                                                                                   0],
                                                                                  ['D loss -real',
                                                                                   'D loss -fake',
                                                                                   'G loss',
                                                                                   'recon loss',
                                                                                   'KLD loss',
                                                                                   'zero'], epoch, i, len(dataloader))
                    elif autoencoder_type == 'AAE':
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [err_discriminator_real.data.mean(),
                                                                                   err_discriminator_fake.data.mean(),
                                                                                   err_generator.data.mean(),
                                                                                   err_recon.data.mean(),
                                                                                   0],
                                                                                  ['D loss -real',
                                                                                   'D loss -fake',
                                                                                   'G loss',
                                                                                   'recon loss',
                                                                                   'zero'], epoch, i, len(dataloader))
                    else:
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [
                                                                                      # z_err.data.mean(),
                                                                                      err_discriminator_real.data.mean(),
                                                                                      err_discriminator_fake.data.mean(),
                                                                                      err_generator.data.mean(),
                                                                                      0],
                                                                                  [
                                                                                      # 'D_z',
                                                                                      'D loss -real',
                                                                                      'D loss -fake',
                                                                                      'G loss',
                                                                                      'zero'],
                                                                                  epoch, i, len(dataloader))
                    grad_line_win_dict = LJY_visualize_tools.draw_lines_to_windict(grad_line_win_dict,
                                                                                   [
                                                                                       discriminator_grad,
                                                                                       generator_grad,
                                                                                       0],
                                                                                   ['D gradient',
                                                                                    'G gradient',
                                                                                    'zero'],
                                                                                   epoch, i, len(dataloader))
                else:
                    if autoencoder_type == 'VAE':
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [
                                                                                      err_recon.data.mean(),
                                                                                      err_KLD.data.mean(),
                                                                                      0],
                                                                                  [
                                                                                      'recon loss',
                                                                                      'KLD loss',
                                                                                      'zero'], epoch, i, len(dataloader))
                    elif autoencoder_type == 'AAE':
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [err_discriminator_real.data.mean(),
                                                                                   err_discriminator_fake.data.mean(),
                                                                                   err_generator.data.mean(),
                                                                                   err_recon.data.mean(),
                                                                                   0],
                                                                                  ['D loss -real',
                                                                                   'D loss -fake',
                                                                                   'G loss',
                                                                                   'recon loss',
                                                                                   'zero'], epoch, i, len(dataloader))
                    else:
                        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                                  [

                                                                                      err.data.mean(),
                                                                                      0],
                                                                                  [
                                                                                      # 'D_z',

                                                                                      'loss',
                                                                                      'zero'],
                                                                                  epoch, i, len(dataloader))

        if options.display_type =='per_epoch':
            if options.dataset == 'MG':
                noise = Variable(torch.FloatTensor(1000, nz)).cuda()
                noise.data.normal_(0, 1)
                generated_fake = decoder(noise)
                MGplot(MGdset, generated_fake, 0, epoch, 0, False)
            if options.intergrationType != 'AEonly':
                if autoencoder_type=='VAE':
                    line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                              [   err_discriminator_real.data.mean(),
                                                                                  err_discriminator_fake.data.mean(),
                                                                                  err_generator.data.mean(),
                                                                                  err_recon.data.mean(),
                                                                                  err_KLD.data.mean(),
                                                                                  0],
                                                                              [   'D loss -real',
                                                                                  'D loss -fake',
                                                                                  'G loss',
                                                                                  'recon loss',
                                                                                  'KLD loss',
                                                                                  'zero'], 0, epoch, 0)
                elif autoencoder_type=='AAE':
                    line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                              [   err_discriminator_real.data.mean(),
                                                                                  err_discriminator_fake.data.mean(),
                                                                                  err_generator.data.mean(),
                                                                                  err_recon.data.mean(),
                                                                                  0],
                                                                              [   'D loss -real',
                                                                                  'D loss -fake',
                                                                                  'G loss',
                                                                                  'recon loss',
                                                                                  'zero'], 0, epoch, 0)
                else:
                    line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                              [
                                                                               #z_err.data.mean(),
                                                                               err_discriminator_real.data.mean(),
                                                                               err_discriminator_fake.data.mean(),
                                                                               err_generator.data.mean(),
                                                                               0],
                                                                              [
                                                                               #'D_z',
                                                                               'D loss -real',
                                                                               'D loss -fake',
                                                                               'G loss',
                                                                               'zero'],
                                                                              0, epoch, 0)
                grad_line_win_dict = LJY_visualize_tools.draw_lines_to_windict(grad_line_win_dict,
                                                                          [
                                                                              discriminator_grad,
                                                                              generator_grad,
                                                                              0],
                                                                          [   'D gradient',
                                                                              'G gradient',
                                                                              'zero'],
                                                                          0, epoch, 0)
            else:
                if autoencoder_type == 'VAE':
                    line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                              [
                                                                               err_recon.data.mean(),
                                                                               err_KLD.data.mean(),
                                                                               0],
                                                                              [
                                                                               'recon loss',
                                                                               'KLD loss',
                                                                               'zero'], 0, epoch, 0)
                else:
                    line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                              [
                                                                                  err.data.mean(),
                                                                                  0],
                                                                              [
                                                                                  # 'D_z',

                                                                                  'loss',
                                                                                  'zero'],
                                                                              0, epoch, 0)

        if options.nz == 2 and visualize_latent:
            #validation visualize

            vis_x = []
            vis_y = []
            vis_label = []
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

                zd = z.data.view(batch_size, nz)
                for i in range(batch_size):
                    vis_x.append(float(zd[i][0]))
                    vis_y.append(float(zd[i][1]))
                    vis_label.append(int(label[i]))
                print("[%d/%d]" % (j, len(val_dataloader)))
            for j in range(int(len(val_dataloader) / 10)):
                for i in range(batch_size):
                    vis_x.append(float(zd.normal_(0, 1)[i][0]))
                    vis_y.append(float(zd.normal_(0, 1)[i][1]))
                    vis_label.append(int(11))

            fig = plt.figure()
            plt.scatter(vis_x, vis_y, c=vis_label, s=2, cmap='rainbow')
            cb = plt.colorbar()
            plt.ylim(-5, 5)
            plt.xlim(-5, 5)
            plt.savefig(os.path.join(validation_path,
                                     "ours_%06d.png" % epoch))
            plt.close()

        # do checkpointing
        if epoch % options.save_tick == 0 or options.save:
            torch.save(encoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch+ep)))
            torch.save(decoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % (epoch+ep)))
            torch.save(discriminator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % (epoch+ep)))
            torch.save(z_discriminator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % (epoch+ep)))
            print(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch+ep)))

def tsne():
    ep = options.pretrainedEpoch
    if ep != 0:
        encoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
        discriminator.load_state_dict(torch.load(
            os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))
        z_discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % ep)))
    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=1, shuffle=False, num_workers=options.workers)

    #vis_x = []
    #vis_y = []
    vis_label = []
    vis = []
    print("Testing Start!")
    for i, (data, label) in enumerate(dataloader, 0):
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

        vis.append(np.asarray(z.data.view(nz)))
        #vis_x.append(z.data.view(nz))
        #vis_y.append(z.data.view(nz))

        vis_label.append(int(label))
        print("[%d/%d]" % (i, len(dataloader)))
    vis = np.asarray(vis)
    z_embedded = TSNE(n_components=2).fit_transform(vis)
    fig = plt.figure()
    plt.scatter(z_embedded[:,0], z_embedded[:,1], c=vis_label, s=2, cmap='tab10')
    cb = plt.colorbar()
    plt.show()


def test(modelname,ep):
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
    for j in range(int(len(dataloader)/10)):
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
    #plt.show()

def visualize_latent_space_2d():
    ep = 10
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    decoder.load_state_dict(
        torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))

    num_range = 20
    x = range(-num_range, num_range)
    y = range(-num_range, num_range)
    image = torch.FloatTensor()
    for i in range(len(x)):
        x_image = torch.FloatTensor()
        for j in range(len(y)):
            z = Variable(torch.FloatTensor(1, 2)).cuda()
            z.data[0][0] = x[i]/(num_range/2)
            z.data[0][1] = y[j]/(num_range/2)
            recon_x = decoder(z.view(z.shape[0], z.shape[1], 1, 1))
            recon_x = unorm(recon_x.data)
            if len(x_image) == 0:
                x_image = recon_x
            else:
                x_image = torch.cat((x_image, recon_x), 2)
        if len(image) == 0:
            image = x_image
        else:
            image = torch.cat((image, x_image), 3)
    img = np.asarray(unorm(image.view(image.shape[2], image.shape[3])))
    plt.imshow(img)
    plt.show()

def visualize_latent_space():
    ep = 10
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
    num_range = 20
    x = range(-num_range, num_range)
    y = range(-num_range, num_range)
    image = torch.FloatTensor()
    for i in range(len(x)):
        x_image = torch.FloatTensor()
        for j in range(len(y)):
            z = Variable(torch.FloatTensor(1, 2)).cuda()
            z.data[0][0] = x[i]/(num_range/2)
            z.data[0][1] = y[j]/(num_range/2)
            recon_x = decoder(z.view(z.shape[0], z.shape[1], 1, 1))
            recon_x = unorm(recon_x.data)
            if len(x_image) == 0:
                x_image = recon_x
            else:
                x_image = torch.cat((x_image, recon_x), 2)
        if len(image) == 0:
            image = x_image
        else:
            image = torch.cat((image, x_image), 3)
    img = np.asarray(unorm(image.view(image.shape[2], image.shape[3])))
    plt.imshow(img)
    plt.show()
def generate():
    num_gen = 10000
    generate_path = os.path.join(os.path.dirname(options.modelOutFolder),
                                   'generated_%s_%s_%s' % (options.dataset, options.intergrationType, options.autoencoderType))
    generate_path = LJY_utils.make_dir(generate_path, allow_duplication=True)

    for ep in range(1,11):
        save_root = "/media/leejeyeol/74B8D3C8B8D38750/Experiment/HMDB_OF/%d"%ep
        LJY_utils.make_dir(save_root)

        decoder.load_state_dict(
            torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
        unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        toimg = transforms.ToPILImage()

        print("Testing Start!")
        for i in range(num_gen):
            noise = Variable(torch.FloatTensor(1, nz)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = decoder(noise.view(1, nz, 1, 1))
            toimg(unorm(generated_fake.data[0]).cpu()).save(save_root+"/%05d.png"%i)
            print('[%d/%d][%d/%d]'%(ep,10,i,num_gen))
def generate_MG():
    MGdset = data_generator()
    # MGdset.random_distribution()
    MGdset.uniform_distribution()
    d_real_data = torch.from_numpy(MGdset.sample(1000))
    plt.figure(figsize=(5, 5))
    plt.scatter(d_real_data[:, 0], d_real_data[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(MGdset.centers[:, 0], MGdset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/original_MG.png')
    plt.close()

    d_real_data = np.asarray(d_real_data)
    bg_color = sns.color_palette('Greens', n_colors=256)[0]
    ax = sns.kdeplot(d_real_data[:, 0], d_real_data[:, 1], shade=True, cmap='Greens', n_levels=20, clip=[[-5, 5]] * 2)
    ax.set_facecolor(bg_color)
    kde = ax.get_figure()
    kde.savefig('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/original_MG_seaborn.png')
    print('done')


if __name__ == "__main__" :
    train()
    #classifier()
    #test('MNIST_AAEGAN',100)
    #tsne()
    #visualize_latent_space_2d()
    #generate()
    #generate_MG()



# Je Yeol. Lee \[T]/
# Jolly Co-operation.tolist()