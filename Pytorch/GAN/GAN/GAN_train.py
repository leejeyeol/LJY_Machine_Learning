import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import GAN.GAN.GAN_model as model
# import custom package
import LJY_utils
import LJY_visualize_tools

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nz', type=int, default=62, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

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
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)


#=======================================================================================================================
# Models
#=======================================================================================================================

# Generator ============================================================================================================
netG = model._netG(ngpu, in_channels=nz)
netG.apply(LJY_utils.weights_init)
if options.netG != '':
    netG.load_state_dict(torch.load(options.netG))
print(netG)

# Discriminator ========================================================================================================
netD = model._netD(ngpu)
netD.apply(LJY_utils.weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)

#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerG = optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=1e-3)



# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
noise = torch.FloatTensor(options.batchSize, nz, 1, 1)

label = torch.FloatTensor(options.batchSize)


if options.cuda:
    netD.cuda()
    netG.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()


# make to variables ====================================================================================================
input = Variable(input)
label = Variable(label,requires_grad = False)
noise = Variable(noise)

win_dict = LJY_visualize_tools.win_dict()
line_win_dict =LJY_visualize_tools.win_dict()
# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (data, target) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================

        optimizerD.zero_grad()

        real_cpu = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        label.data.resize_(batch_size).fill_(1)
        real_label = label.clone()
        label.data.resize_(batch_size).fill_(0)
        fake_label = label.clone()

        outputD = netD(input)
        errD_real = criterion_D(outputD, real_label)
        #errD_real.backward()
        D_x = outputD.data.mean()   # for visualize

        # generate noise    ============================================================================================
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        #train with fake data   ========================================================================================
        fake = netG(noise)
        outputD = netD(fake.detach())
        errD_fake = criterion_D(outputD, fake_label)
        #errD_fake.backward()
        D_G_z1 = outputD.data.mean()

        errD = errD_fake+errD_real
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network and Q network
        ###########################
        optimizerG.zero_grad()

        #fake = Variable(fake.data, requires_grad = True)
        outputD = netD(fake)
        errG = criterion_G(outputD, real_label)
        errG.backward()
        D_G_z2 = outputD.data.mean()
        optimizerG.step()

        #visualize
        print('[%d/%d][%d/%d] Loss_D: %.4f(1:%.4f 2:%.4f) Loss_G: %.4f     D(x): %.4f D(G(z)): %.4f | %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0],errD_real.mean(),errD_fake.mean(), errG.data[0],  D_x, D_G_z1, D_G_z2))
        if True:
            testImage = fake.data[0]
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["MNIST GAN"])
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [errD.data.mean(), errG.data.mean(), D_x,
                                                                       D_G_z1],
                                                                      ['lossD', 'lossG', 'real is?', 'fake is?'], epoch, i,
                                                                      len(dataloader))

    # do checkpointing
    if epoch % 10 == 0 :
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/