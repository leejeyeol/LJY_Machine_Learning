import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import GAN.SandwichGAN.SandwichGAN_model as model
import GAN.SandwichGAN.SandwichGAN_dataloader as dset
# import custom package
import LJY_utils

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='HMDB51', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/HMDB51/middle_block', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./output(for_test)', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='InfoGAN', help='Model name')
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

IsSave = False

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
    dset.SandwichGAN_Dataloader(options.dataroot,centered=True),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers, drop_last=False)
# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)


#=======================================================================================================================
# Models
#=======================================================================================================================

# Generator ============================================================================================================
netG = model._netG(ngpu)
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

#Q_Influence = 1.0
# todo add betas=(0.5, 0.999),
#optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerD = optim.SGD(netD.parameters(),lr = 0.0002, momentum=0.9)
#optimizerG = optim.SGD(netD.parameters(),lr = 0.0002, momentum=0.9)
optimizerG = optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=0.02)


# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
bread = torch.FloatTensor(options.batchSize, 6, options.imageSize, options.imageSize)
sandwich = torch.FloatTensor(options.batchSize, 9, options.imageSize, options.imageSize)
label = torch.FloatTensor(options.batchSize,1,1,1)
real_label = 1
fake_label = 0

if options.cuda:
    netD.cuda()
    netG.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    input, label = input.cuda(), label.cuda()
    bread, sandwich = bread.cuda(), sandwich.cuda()


# make to variables ====================================================================================================
input = Variable(input)
label = Variable(label)
bread = Variable(bread)
sandwich = Variable(sandwich)


# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (pre_frame, real_mid_frame, nxt_frame,_) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        pre_frame = pre_frame.cuda()
        real_mid_frame = real_mid_frame.cuda()
        nxt_frame = nxt_frame.cuda()

        optimizerD.zero_grad()
        optimizerG.zero_grad()


        batch_size = pre_frame.size(0)
        input.data.resize_(pre_frame.size()).copy_(pre_frame)
        label.data.fill_(real_label)

        pre_and_nxt = torch.cat((pre_frame, nxt_frame), 1)
        bread.data = pre_and_nxt.cuda()
        fake_mid_frame = netG(bread)

        real_sandwich = torch.cat((pre_frame, real_mid_frame, nxt_frame), 1)
        fake_sandwich = torch.cat((pre_frame, fake_mid_frame.data, nxt_frame), 1)

        sandwich.data = real_sandwich
        outputD = netD(sandwich)
        if outputD.data.max() <0 or outputD.data.min()>1:
            print(1)
        print(outputD.data.max())
        print(outputD.data.min())
        errD_real = criterion_D(outputD, label)
        errD_real.backward()

        vis_D_x = outputD.data.mean()   # for visualize

        label.data.fill_(fake_label)
        sandwich.data = fake_sandwich
        outputD = netD(sandwich.detach())
        errD_fake = criterion_D(outputD, label)
        errD_fake.backward()

        vis_D_G_z1 = outputD.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network and Q network
        ###########################
        optimizerG.zero_grad()
        label.data.fill_(real_label)

        # fake labels are real for generator cost
        sandwich.data = fake_sandwich
        outputD = netD(sandwich)

        errG = criterion_G(outputD, label)
        errG.backward(retain_graph=True)
        vis_D_G_z2 = outputD.data.mean()
        optimizerG.step()

        #visualize
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f   D(x): %.4f(to 1) D(G(z)): %.4f(to 0) | %.4f(fake=real?)'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0], errG.data[0], vis_D_x, vis_D_G_z1, vis_D_G_z2))

        #if i == len(dataloader)-1:
        if True:
            testImage = torch.cat((pre_frame[0],real_mid_frame[0],fake_mid_frame.data[0],nxt_frame[0]),2)
            vutils.save_image(testImage,
                    '%s/%d_test_samples.png' % (options.outf,i),
                    normalize=True)

    # do checkpointing
    if IsSave == True:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation