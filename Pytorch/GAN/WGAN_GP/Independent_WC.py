import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import Pytorch.GAN.WGAN_GP.WGAN_GP_model as model
import Pytorch.GAN.WGAN_GP.WGAN_GP_dataloader as dset
import numpy as np
# import custom package
import LJY_utils
import LJY_visualize_tools

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png', help='path to dataset')
parser.add_argument('--fold', type=int,default=None, help = 'fold number')
parser.add_argument('--fold_dataroot', default='',help='Proprocessing/fold_divider.py')

parser.add_argument('--netG', default='/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/W_Critic/Base_CelebA_decoder_9.pth', help="path of Generator networks. pretrained Generator")
parser.add_argument('--netD', default='/home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/WGAN_GP/output/WGANGP_netD_epoch_36.pth', help="path of Discriminator networks. Wasserstein Critic")
parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False.')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=64, help='dimension of noise.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)




def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = LJY_utils.cuda(torch.rand(shape))
    z = x + alpha * (y - x)

    # gradient penalty
    z = LJY_utils.cuda(Variable(z, requires_grad=True))
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=LJY_utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp


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


# ======================================================================================================================
# Data and Parameters
# ======================================================================================================================
display = options.display
dataset= options.dataset
batch_size = options.batchSize
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)
nc = int(options.nc)
# CelebA call and load   ===============================================================================================


transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
if options.fold is None:
    dataloader = torch.utils.data.DataLoader(dset.Dataloader(options.dataroot, transform,type='val'),
                                             batch_size=options.batchSize, shuffle=True, num_workers=options.workers,drop_last=False)

else:
    dataloader = torch.utils.data.DataLoader(dset.fold_Dataloader(options.fold, options.fold_dataroot, transform, type='train'),
                                             batch_size=options.batchSize, shuffle=True, num_workers=options.workers,drop_last=False)


unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

# ======================================================================================================================
# Models
# ======================================================================================================================

# Generator ============================================================================================================
netG = model.decoder64x64(num_in_channels=nc, z_size=nz, num_filters=64)
netG.apply(LJY_utils.weights_init)
if options.netG != '':
    netG.load_state_dict(torch.load(options.netG))
print(netG)

# Discriminator ========================================================================================================
netD = model.DiscriminatorWGANGP(nc)
netD.apply(LJY_utils.weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)


if options.cuda:
    netD.cuda()
    netG.cuda()
wd_list = []
for i, data in enumerate(dataloader, 0):
    test_data = Variable(data, requires_grad=False).cuda()
    outputD_real = netD(test_data)

    noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1), requires_grad=False).cuda()
    noise.data.normal_(0, 1)
    fake_data = netG(noise)
    outputD_fake = netD(fake_data)
    wd = (torch.mean(outputD_real) - torch.mean(outputD_fake))
    wd_list.append(float(wd.data))
    print('[%d/%d]'%(i,len(dataloader)))
np_wd_list = np.asarray(wd_list)
np.save('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/W_Critic/%s_%d_%d.npy' % ("Base", 9, 36), np_wd_list)

# Je Yeol. Lee \[T]/