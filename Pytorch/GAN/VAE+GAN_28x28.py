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
import seaborn
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
parser.add_argument('--dataset', default='HMDB51_224', help='what is dataset? MG : Mixtures of Gaussian', choices=['CelebA', 'MNIST', 'MG'])
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/flow/flow', help='path to dataset')

parser.add_argument('--autoencoderType', default='AAE', help='additional autoencoder type. "GAN" use DCGAN only', choices=['AE', 'VAE', 'AAE', 'GAN'])
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks. '0' is training from scratch.")
parser.add_argument('--pretrainedModelName', default='HMDB_OF', help="path of Encoder networks.")
parser.add_argument('--modelOutFolder', default='./pretrained_model', help="folder to model checkpoints")
parser.add_argument('--resultOutFolder', default='./results', help="folder to test results")
parser.add_argument('--save_tick', type=int, default=1, help='save tick')
parser.add_argument('--display_type', default='per_iter', help='displat tick',choices=['per_epoch', 'per_iter'])

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--save', default=False, help='save options. default:False. NOT IMPLEMENTED')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=15000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=256, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)
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
class unbiased_MNIST(dset.MNIST):
    def __init__(self, root,  train, download, transform):
        super().__init__(root=root,  train=train, download=download, transform=transform)
        idx_per_label = [[] for _ in range(10)]
        label_list = list(self.train_labels)
        for i in range(len(label_list)):
            idx_per_label[label_list[i]].append(i)
        random.seed(1)
        # 0 : 5923 => 5923
        # container
        unbiased_train_data = self.train_data[0].view(1, 28, 28)
        unbiased_labels = [self.train_labels[0]]

        # sampling data
        for idx in idx_per_label[0]:
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])

        self.train_labes = torch.LongTensor(unbiased_labels)
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

def MGplot(MGdset, save_path, points, epoch, iteration, total_iter):
    idx = epoch * total_iter + iteration
    plt.figure(figsize=(5,5))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='r', alpha=0.5)
    plt.scatter(MGdset.centers[:, 0], MGdset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig(os.path.join(save_path, "MG_result","powerful_vae_%06d.png" % idx))
    plt.close()

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

def gaussian_noise(ins, is_training, mean=0, stddev=1,prob = 0.9):
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

def Variational_loss(input, target, mu, logvar):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    return recon_loss, KLD_loss


class encoder224x224(nn.Module):
    '''encoder'''

    def __init__(self, nz, nc, type = 'VAE', large=False):
        super(encoder224x224, self).__init__()

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

        return h

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

class discriminator(nn.Module):
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

class z_discriminator(nn.Module):
    def __init__(self, N=1000, z_dim=120):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, N),
            nn.Dropout(0.2),
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

class MG_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MG_decoder, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        #self.activation_fn = F.tanh
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)

class MG_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, type):
        super(MG_encoder, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.relu
        self.final_activation_fn = F.sigmoid
        self.fc_mu = nn.Linear(output_size, output_size)
        self.fc_sig = nn.Linear(output_size, output_size)
        self.type = type

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            # AE
            x = self.activation_fn(self.map1(x))
            x = self.activation_fn(self.map2(x))
            z = self.final_activation_fn(self.map3(x))
            return z
        elif self.type == 'VAE':
            # VAE
            x = self.activation_fn(self.map1(x))
            x = self.activation_fn(self.map2(x))
            z_ = self.map3(x)
            mu = self.final_activation_fn(self.fc_mu(z_))
            logvar = self.final_activation_fn((self.fc_sig(z_)))
            return mu, logvar

class MG_discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MG_discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.relu
        self.final_activation_fn = F.sigmoid

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.final_activation_fn(self.map3(x))

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
if options.dataset == 'MNIST':
    encoder = encoder(num_in_channels=1, z_size=nz, num_filters=64 ,type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    decoder = decoder(num_in_channels=1, z_size=nz, num_filters=64)
    decoder.apply(LJY_utils.weights_init)
    print(decoder)

    discriminator = discriminator(num_in_channels=1, num_filters=64)
    discriminator.apply(LJY_utils.weights_init)
    print(discriminator)

elif options.dataset == 'CelebA':
    encoder = encoder64x64(num_in_channels=3, z_size=nz, num_filters=64,type=autoencoder_type)
    encoder.apply(LJY_utils.weights_init)
    print(encoder)

    decoder = decoder64x64(num_in_channels=3, z_size=nz, num_filters=64)
    decoder.apply(LJY_utils.weights_init)
    print(decoder)

    discriminator = discriminator64x64(num_in_channels=3, num_filters=64)
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
z_discriminator = z_discriminator(N=500, z_dim=nz)
z_discriminator.apply(LJY_utils.weights_init)
print(z_discriminator)

#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average=False)

# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
#optimizerDiscriminator = optim.RMSprop(discriminator.parameters(), lr=2e-4)
#optimizerDiscriminator = optim.SGD(discriminator.parameters(), lr=2e-4)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)
optimizer_z_Discriminator = optim.Adam(z_discriminator.parameters(), betas=(0.5, 0.999), lr=2e-4)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    z_discriminator.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()


# training start
def train():
    ep = options.pretrainedEpoch
    if ep != 0:
        #encoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
        discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))
        #z_discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % ep)))
    if options.dataset == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            dset.MNIST(root='../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        '''
        dataloader = torch.utils.data.DataLoader(
            unbiased_MNIST(root='../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        '''
    elif options.dataset == 'CelebA':
        dataloader = torch.utils.data.DataLoader(
            custom_Dataloader(path=options.dataroot,
                              transform=transforms.Compose([
                                  transforms.CenterCrop(150),
                                  transforms.Scale(64),
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
        dataloader = torch.utils.data.DataLoader(MG_Dataloader(options.epoch, MGdset),batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    grad_line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")

    alpha = 0.5


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

            if autoencoder_type == "VAE":
                optimizerE.zero_grad()
                optimizerD.zero_grad()
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_recon, err_KLD = Variational_loss(x_recon, input.detach(), mu, logvar)
                err = (alpha * err_recon) + err_KLD
                err.backward(retain_graph=True)
                generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                optimizerE.step()
                optimizerD.step()
            elif autoencoder_type == 'AE' or autoencoder_type == 'AAE':
                optimizerE.zero_grad()
                optimizerD.zero_grad()
                z = encoder(input)
                x_recon = decoder(z)
                err = MSE_loss(x_recon, input.detach())
                err = alpha * err
                err.backward(retain_graph=True)
                generator_grad_AE = LJY_utils.torch_model_gradient(decoder.parameters())
                optimizerE.step()
                optimizerD.step()

            if autoencoder_type == 'AAE':
                optimizer_z_Discriminator.zero_grad()
                optimizerE.zero_grad()
                z = encoder(input)
                noise_z = Variable(torch.FloatTensor(z.view(z.shape[0], z.shape[1]).shape).normal_(0, 1).cuda())
                z_d_real = z_discriminator(noise_z)
                z_d_fake = z_discriminator(z.view(z.shape[0], z.shape[1]))
                z_err = (BCE_loss(z_d_real, real_label) + BCE_loss(z_d_fake, fake_label))
                z_err.backward(retain_graph=True)
                optimizer_z_Discriminator.step()

                generator_err = BCE_loss(z_d_fake, real_label)
                generator_err.backward(retain_graph=True)
                generator_grad_AE += LJY_utils.torch_model_gradient(decoder.parameters())
                optimizerE.step()

            # discriminator training =======================================================================================
            optimizerDiscriminator.zero_grad()
            d_real = discriminator(disc_input)
            noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = decoder(noise.view(batch_size, nz, 1, 1))
            d_fake = discriminator(generated_fake)

            if autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                recon_real = decoder(z)
                d_fake_recon = discriminator(recon_real)

            elif autoencoder_type == 'AE' or autoencoder_type == 'AAE':
                z = encoder(disc_input)
                recon_real = decoder(z)
                d_fake_recon = discriminator(recon_real)

            balance_coef = torch.cat((d_fake, d_real), 0).mean()
            err_discriminator_real = BCE_loss(d_real.view(batch_size), real_label.view(batch_size))
            err_discriminator_fake = BCE_loss(d_fake.view(batch_size), fake_label.view(batch_size))

            err_discriminator_origin = err_discriminator_real + err_discriminator_fake#+ BCE_loss(d_fake_recon, fake_label))
                                       #+ BCE_loss(balance_coef, Variable(torch.FloatTensor(balance_coef.shape).fill_(0.5).cuda()))
            err_discriminator = err_discriminator_origin
            err_discriminator.backward(retain_graph=True)
            discriminator_grad = LJY_utils.torch_model_gradient(discriminator.parameters())
            optimizerDiscriminator.step()

            optimizerD.zero_grad()
            noise = Variable(torch.FloatTensor(batch_size, nz ,1 ,1)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = decoder(noise)
            d_fake_2 = discriminator(generated_fake)
            err_generator = BCE_loss(d_fake_2.view(batch_size), real_label.view(batch_size)) #+ BCE_loss(d_fake_recon, real_label))
            err_generator = (1-alpha) * err_generator
            err_generator.backward(retain_graph=True)

            generator_grad = LJY_utils.torch_model_gradient(decoder.parameters())
            if autoencoder_type == 'GAN':
                generator_grad_AE = generator_grad

            optimizerD.step()
             #visualize
            print('[%d/%d][%d/%d] recon_Loss: GAN  d_real: %.4f d_fake: %.4f Balance : %.2f alpha : %.2f'
                  % (epoch, options.epoch, i, len(dataloader), d_real.data.mean(), d_fake_2.data.mean(), balance_coef.data.mean(),alpha))
            if options.display:
                if autoencoder_type == 'GAN':
                    testImage = torch.cat((unorm(input.data[0]), unorm(generated_fake.data[0])), 2)
                else:
                    testImage = torch.cat((unorm(input.data[0]), unorm(x_recon.data[0]), unorm(generated_fake.data[0])), 2)
                win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            if options.display_type == 'per_iter':
                if options.dataset == 'MG':
                    noise = Variable(torch.FloatTensor(1000, nz)).cuda()
                    noise.data.normal_(0, 1)
                    generated_fake = decoder(noise)
                    MGplot(MGdset, options.modelOutFolder, generated_fake, epoch, i, len(dataloader))
                line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                          [
                                                                              err_discriminator_real.data.mean(),
                                                                              err_discriminator_fake.data.mean(),
                                                                              err_generator.data.mean(),
                                                                              0],
                                                                          [
                                                                              'D loss -real',
                                                                              'D loss -fake',
                                                                              'G loss',
                                                                              'zero'],
                                                                          epoch, i, len(dataloader))
                grad_line_win_dict = LJY_visualize_tools.draw_lines_to_windict(grad_line_win_dict,
                                                                               [
                                                                                   discriminator_grad,
                                                                                   generator_grad_AE,
                                                                                   generator_grad,
                                                                                   0],
                                                                               ['D gradient',
                                                                                'G gradient_AE',
                                                                                'G gradient',
                                                                                'zero'],
                                                                               epoch, i, len(dataloader))

        if options.display_type =='per_epoch':
            if options.dataset == 'MG':
                noise = Variable(torch.FloatTensor(1000, nz)).cuda()
                noise.data.normal_(0, 1)
                generated_fake = decoder(noise)
                MGplot(MGdset, options.modelOutFolder, generated_fake, 0, epoch, 0)
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
                                                                          generator_grad_AE,
                                                                          generator_grad,
                                                                          0],
                                                                      [   'D gradient',
                                                                          'G gradient_AE',
                                                                          'G gradient',
                                                                          'zero'],
                                                                      0, epoch, 0)



        # do checkpointing

        if epoch % options.save_tick == 0 or options.save:
            #torch.save(encoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch+ep)))
            torch.save(decoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % (epoch+ep)))
            torch.save(discriminator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % (epoch+ep)))
            #torch.save(z_discriminator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % (epoch+ep)))
            print(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch+ep)))

def tsne():
    ep = 1000
    encoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_encoder_epoch_%d.pth")%ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_decoder_epoch_%d.pth")%ep))
    discriminator.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_discriminator_epoch_%d.pth")%ep))

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


def test():
    ep = 350

    encoder.load_state_dict(torch.load(os.path.join(options.outf, "GAN_encoder_epoch_%d.pth") % ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "GAN_decoder_epoch_%d.pth") % ep))
    discriminator.load_state_dict(
        torch.load(os.path.join(options.outf, "GAN_discriminator_epoch_%d.pth") % ep))

    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=1, shuffle=False, num_workers=options.workers)

    vis_x = []
    vis_y = []
    vis_label = []
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

        vis_x.append(z.data.view(nz)[0])
        vis_y.append(z.data.view(nz)[1])

        vis_label.append(int(label))
        print("[%d/%d]" % (i, len(dataloader)))
    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=vis_label, s=2, cmap='tab10')
    cb = plt.colorbar()
    plt.show()

def visualize_latent_space_2d():
    ep = 360
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
    ep = 50
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "GAN_decoder_epoch_%d.pth") % ep))
    noise = Variable(torch.FloatTensor(4, nz, 1, 1)).cuda()
    noise.data.normal_(0, 1)
    generated_fake = decoder(noise)
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


if __name__ == "__main__" :
    train()
    #test()
    #tsne()
    #visualize_latent_space_2d()


# Je Yeol. Lee \[T]/
# Jolly Co-operation.tolist()