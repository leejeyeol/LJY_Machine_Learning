# -*- coding:utf-8 -*-
# 한글 주석 가능하게
import torch.utils.data as ud
import argparse
import os
import random
import glob as glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

import LJY_utils
import LJY_visualize_tools

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--autoencoderType', default='AE', help='AE, VAE')

parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=(28, 28), help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=120, help='number of input channel.')

parser.add_argument('--pretrainedModelName', default='autoencoder', help="path of Encoder networks.")
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks.")

parser.add_argument('--modelOutFolder', default='../pretrained_model', help="folder to model checkpoints")
parser.add_argument('--resultOutFolder', default='../results', help="folder to test results")
parser.add_argument('--save_tick', type=int, default=10, help='save tick')
parser.add_argument('--visualize_type', default='per_epoch', help='per_epoch, per_iter')
parser.add_argument('--visualize', default=False, help='display options. default:False. NOT IMPLEMENTED')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=12000, help='number of epochs to train for')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)

def Variational_loss(input, target, mu, logvar):
    recon_loss = VAE_recon_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    return recon_loss + KLD_loss

class Custom_Image_DataLoader(torch.utils.data.Dataset):
    def __init__(self, path, transform, size=(28, 28), type='train'):
        super().__init__()
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path
        self.size = size
        if self.type == 'train':
            self.file_paths = glob.glob(os.path.join(self.base_path, 'train', '*'))
        elif self.type == 'val':
            self.file_paths = glob.glob(os.path.join(self.base_path, 'val', '*'))
        elif self.type == 'test':
            self.file_paths = glob.glob(os.path.join(self.base_path, 'test', '*'))

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.resize(self.size)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


class encoder_fcn(nn.Module):
    def __init__(self, x_dim=784, N=1000, z_dim=120, type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, N),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, N),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, z_dim)
        )
        self.fc_mu = nn.Linear(z_dim, z_dim, 1)
        self.fc_sig = nn.Linear(z_dim, z_dim, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE':
            #AE
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

class decoder_fcn(nn.Module):
    def __init__(self, x_dim=784, N=1000, z_dim=120):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, N),
            nn.Dropout(0.25),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, x_dim)
        )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

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

# output folder validation check
try:
    os.makedirs(options.outf)
except OSError:
    pass

# seed configuration
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)
if options.cuda:
    torch.cuda.manual_seed(options.seed)

# cuda(GPU 사용) 설정
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(options.ngpu)
nz = int(options.nz)
autoenocder_type = options.autoencoderType

encoder = encoder_fcn(784, 1000, nz, autoenocder_type)
encoder.apply(LJY_utils.weights_init)
if options.netE != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

decoder = decoder_fcn(784, 1000, nz)
decoder.apply(LJY_utils.weights_init)
if options.netD != '':
    decoder.load_state_dict(torch.load(options.netD))
print(decoder)

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
VAE_recon_loss = nn.MSELoss()

optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    VAE_recon_loss.cuda()

# training start
def train():
    ep = options.pretrainedEpoch
    if ep != 0:
        encoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
        decoder.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))

    dataloader = torch.utils.data.DataLoader(
        Custom_Image_DataLoader(path='/media/leejeyeol/74B8D3C8B8D38750/Data/example_test',
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]), size=options.imageSize, type='train'),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    # visualize를 위한 옵션
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5,), std=(0.5,))
    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()

    print("Training Start!")
    for epoch in range(options.iteration):
        for i, data in enumerate(dataloader, 0):

            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            #vectorize
            input = Variable(real_cpu.view(real_cpu.size(0), -1)).cuda()

            if autoenocder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err = MSE_loss(x_recon, input.detach())
            elif autoenocder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_variational = Variational_loss(x_recon, input.detach(), mu, logvar)
                err_variational.backward(retain_graph=True)

            optimizerE.step()
            optimizerD.step()

            #visualize
            print('[%d/%d][%d/%d] MSE_Loss: %.4f'% (epoch, options.iteration, i, len(dataloader), err_variational.data.mean()))
            testImage = torch.cat((unorm(input.data[0].view(1,28,28)), unorm(x_recon.data[0].view(1,28,28))), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_variational.data.mean(), 0],
                                                                      ['loss_recon_x', 'zero'],
                                                                      epoch, i, len(dataloader))



        # do checkpointing
        if epoch % 1 == 0:
            torch.save(encoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep))
            torch.save(decoder.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep))

def test():
    ep = 1000
    encoder.load_state_dict(torch.load('%s/encoder_epoch_%d.pth' % (options.outf, ep)))
    decoder.load_state_dict(torch.load('%s/decoder_epoch_%d.pth' % (options.outf, ep)))


    dataloader = torch.utils.data.DataLoader(
        Custom_Image_DataLoader(path='/media/leejeyeol/74B8D3C8B8D38750/Data/example_test',
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))
           ]),size=(28,28),type='test'),
        batch_size=1, shuffle=False, num_workers=options.workers)
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5,), std=(0.5,))

    print("Training Start!")
    for i, (data, target) in enumerate(dataloader, 0):
        real_cpu = data
        # vectorize
        input = Variable(real_cpu.view(real_cpu.size(0), -1)).cuda()
        target = Variable(target.view(target.size(0), -1)).cuda()

        mu, logvar = encoder(input)
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
        z = eps.mul(std).add_(mu)

        x_recon = decoder(z)
        testImage =unorm(x_recon.data)
        toimg = transforms.ToPILImage()
        toimg(testImage.view(1,28,28).cpu()).save("/media/leejeyeol/74B8D3C8B8D38750/Data/example_test/result_img" + "/%05d.png" % i)
        print(i)


#train()
test()



# Je Yeol. Lee \[T]/
# Jolly Co-operation