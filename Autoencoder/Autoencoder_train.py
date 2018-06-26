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


import Autoencoder.VGG16_lossnet as vgg
from torchvision import models

import Autoencoder.Autoencoder_model as model
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
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=62, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)



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
    dset.MNIST('../MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
'''
dataloader = torch.utils.data.DataLoader(
    dset.CIFAR10('../CIFAR10', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Scale(224),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)

#=======================================================================================================================
# Models
#=======================================================================================================================



# Generator ============================================================================================================
encoder = model.encoder(ngpu)
encoder.apply(LJY_utils.weights_init)
if options.netG != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

# Discriminator ========================================================================================================
decoder = model.decoder(ngpu)
decoder.apply(LJY_utils.weights_init)
if options.netD != '':
    decoder.load_state_dict(torch.load(options.netD))
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
MSE_loss = nn.MSELoss()
def Perceptual_loss(input, target):
    feature_input = loss_net(input)
    feature_target = loss_net(target)
    loss = 0
    for i in range(4):
        loss = loss + MSE_loss(feature_input[i], feature_target[i])
    return loss/4
# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)



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


# training start
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

        ddddd=vggnet(input)

        z= encoder(input)
        x_recon= decoder(z)

        err_mse = MSE_loss(x_recon, input.detach())
        err_mse.backward(retain_graph=True)
        #err_perceptual = Perceptual_loss(x_recon, input.detach())
        #err_perceptual.backward(retain_graph=True)


        optimizerE.step()
        optimizerD.step()

        #visualize
        print('[%d/%d][%d/%d] Loss: %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 err_mse.data.mean()))
        testImage = torch.cat((input.data[0], x_recon.data[0]), 2)
        win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])
        line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict, [err_mse.data.mean(),0], ['loss_recon_x','zero'], epoch, i, len(dataloader))

    # do checkpointing
    '''
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))
    torch.save(netQ.state_dict(), '%s/netQ_epoch_%d.pth' % (options.outf, epoch))
    '''


# Je Yeol. Lee \[T]/
# Jolly Co-operation