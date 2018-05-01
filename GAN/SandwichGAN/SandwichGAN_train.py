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
import LJY_visualize_tools

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
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

netG_pre = model._encoder(ngpu)
netG_pre.apply(LJY_utils.weights_init)
if options.netG != '':
    netG_pre.load_state_dict(torch.load(options.netG))
print(netG_pre)

netG_nxt = model._encoder(ngpu)
netG_nxt.apply(LJY_utils.weights_init)
if options.netG != '':
    netG_nxt.load_state_dict(torch.load(options.netG))
print(netG_nxt)

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
optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
#optimizerD = optim.SGD(netD.parameters(), lr=0.0002, momentum=0.9)
#optimizerD = optim.RMSprop(netD.parameters(), lr=5e-5)
#optimizerG = optim.SGD(netD.parameters(),lr = 0.0002, momentum=0.9)
optimizerG = optim.Adam(list(netG_pre.parameters())+list(netG_nxt.parameters())+list(netG_pre.parameters()), betas=(0.5, 0.999), lr=0.02)
#optimizerG = optim.RMSprop(netG.parameters(), lr=5e-5)



# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
bread_pre = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
bread_nxt = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
label = torch.FloatTensor(options.batchSize,1,1,1)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1


if options.cuda:
    netD.cuda()
    netG_pre.cuda()
    netG_nxt.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    input, label = input.cuda(), label.cuda()
    bread_pre, bread_nxt = bread_pre.cuda(), bread_nxt.cuda()
    one, mone = one.cuda(), mone.cuda()


# make to variables ====================================================================================================
input = Variable(input)
label = Variable(label)

bread_pre = Variable(bread_pre)
bread_nxt = Variable(bread_nxt)

win_dict = LJY_visualize_tools.win_dict()
# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (pre_frame, real_mid_frame, nxt_frame, _) in enumerate(dataloader, 0):
        if i % 10 == 0:
            diter = 100
        else:
            diter = 1
        for _ in range(diter):
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

            bread_pre.data = pre_frame.cuda()
            bread_nxt.data = nxt_frame.cuda()

            output_pre = netG_pre(bread_pre)
            output_nxt = netG_nxt(bread_nxt)
            pre_and_nxt = torch.cat((output_pre, output_nxt), 1).cuda()

            fake_mid_frame = netG(pre_and_nxt)

            real_sandwich = torch.cat((pre_frame, real_mid_frame, nxt_frame), 1)
            fake_sandwich = torch.cat((pre_frame, fake_mid_frame.data, nxt_frame), 1)

            real_sandwich = Variable(real_sandwich)
            outputD = netD(real_sandwich)
            errD_real = -torch.mean(outputD)
            vis_D_x = errD_real.data.mean()   # for visualize

            # train with fake data  ========================================================================================
            fake_sandwich = Variable(fake_sandwich)
            outputD = netD(fake_sandwich.detach())
            errD_fake = torch.mean(outputD)
            vis_D_G_z1 = errD_fake.data.mean()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

        ############################
        # (2) Update G network
        ###########################
        optimizerG.zero_grad()

        pre_and_nxt = torch.cat((output_pre, output_nxt), 1).cuda()
        fake_mid_frame = netG(pre_and_nxt)
        fake_sandwich = torch.cat((pre_frame, fake_mid_frame.data, nxt_frame), 1)
        fake_sandwich = Variable(fake_sandwich)
        outputG = netD(fake_sandwich)
        errG = -torch.mean(outputG)
        #errG = criterion_G(outputG, label)
        errG.backward(one)
        vis_D_G_z2 = errG.data.mean()
        optimizerG.step()


        #visualize
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f   D(x): %.4f(to 1) D(G(z)): %.4f(to 0) | %.4f(fake=real?)'
             % (epoch, options.iteration, i, len(dataloader),
                errD.data[0], errG.data[0], vis_D_x, vis_D_G_z1, vis_D_G_z2))

        #if i == len(dataloader)-1:
        if True:
            testImage = torch.cat((pre_frame[0],real_mid_frame[0],fake_mid_frame.data[0],nxt_frame[0]),2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Sandwich GAN"])

            '''
            vutils.save_image(testImage,
                    '%s/%d_test_samples.png' % (options.outf,i),
                    normalize=True)
            '''

    # do checkpointing
    if IsSave == True:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation