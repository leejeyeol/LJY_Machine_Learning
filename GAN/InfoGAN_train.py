import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
# import custom package
import LJY_utils
import GAN.InfoGAN_model as model



#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./InfoGAN', help="folder to output images and model checkpoints")

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

ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)
nc = int(options.nc)
nconC = 2
ncatC = 10

#=======================================================================================================================
# Models
#=======================================================================================================================

# Generator ============================================================================================================
netG = model._netG(ngpu, in_channels=nz+nconC+ncatC)
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

# Auxiliary distribution ===============================================================================================
netQ = model._netQ(ngpu, ncatC=ncatC, nconC=nconC)
netQ.apply(LJY_utils.weights_init)
if options.netQ != '':
    netQ.load_state_dict(torch.load(options.netQ))
print(netQ)


#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()
criterion_cat = nn.BCELoss()
criterion_con = nn.MSELoss()

# setu
# p optimizer   ====================================================================================================
Q_Influence = 1.0
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerG = optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=1e-3)
optimizerQ = optim.Adam(netQ.parameters(), betas=(0.5, 0.999), lr=2e-4)


# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
final_noise = torch.FloatTensor(options.batchSize, nz+nconC+ncatC, 1, 1)
noise = torch.FloatTensor(options.batchSize, nz, 1, 1)

noise_c1 = torch.FloatTensor(options.batchSize, 1, 1, 1)
noise_c2 = torch.FloatTensor(options.batchSize, 1, 1, 1)
onehot_c = torch.FloatTensor(options.batchSize, 10)
onehot_c_f = torch.FloatTensor(options.batchSize, 10, 1, 1)



# for check   ==========================================================================================================
fixed_noise = torch.FloatTensor(options.batchSize, nz+nconC+ncatC, 1, 1).normal_(0, 1)

label = torch.FloatTensor(options.batchSize)
real_label = 1
fake_label = 0

if options.cuda:
    netD.cuda()
    netG.cuda()
    netQ.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    criterion_cat.cuda()
    criterion_con.cuda()
    input, label = input.cuda(), label.cuda()
    final_noise, noise, fixed_noise, noise_c1, noise_c2, onehot_c, onehot_c_f = final_noise.cuda(), noise.cuda(), \
                                                                    fixed_noise.cuda(), noise_c1.cuda(),\
                                                                    noise_c2.cuda(), onehot_c.cuda(), onehot_c_f.cuda()



# make to variables ====================================================================================================
input = Variable(input)
label = Variable(label)

final_noise = Variable(final_noise)

noise = Variable(noise)
noise_c1 = Variable(noise_c1)
noise_c2 = Variable(noise_c2)
onehot_c = Variable(onehot_c)
onehot_c_f = Variable(onehot_c_f)
fixed_noise = Variable(fixed_noise)



# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real data  ========================================================================================
        optimizerD.zero_grad()
        optimizerQ.zero_grad()
        optimizerG.zero_grad()

        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        outputD, _ = netD(input)
        errD_real = criterion_D(outputD, label)
        errD_real.backward()

        D_x = outputD.data.mean()   # for visualize

        # generate noise    ============================================================================================
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        noise_c1.data.resize_(batch_size, 1, 1, 1)
        nn.init.uniform(noise_c1, -1, 1)
        noise_c2.data.resize_(batch_size, 1, 1, 1)
        nn.init.uniform(noise_c2, -1, 1)

        onehot_c.data = LJY_utils.one_hot((batch_size, 10), torch.LongTensor([random.randrange(0, 10) for i in
                                                                              range(batch_size)]).view(-1, 1)).cuda()

        onehot_c.data = onehot_c.detach().data.resize_(batch_size, 10, 1, 1).float()

        final_noise = torch.cat((noise, noise_c1, noise_c2, onehot_c), 1)

        #train with fake data   ========================================================================================
        fake = netG(final_noise)
        label.data.fill_(fake_label)

        outputD, _ = netD(fake.detach())
        errD_fake = criterion_D(outputD, label)
        errD_fake.backward()

        D_G_z1 = outputD.data.mean()

        errD = errD_real + errD_fake
       # errD.backward()
        optimizerD.step()

        #just optimize q???

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        outputD, midQ = netD(fake)
        errG = criterion_G(outputD, label)
        errG.backward(retain_variables=True)

        c_cat, c_cont = netQ(midQ)
        errQ_cat = criterion_cat(c_cat, onehot_c)
        errQ_con = criterion_con(c_cont, torch.cat((noise_c1, noise_c2), 1))

        hook = errQ_cat.register_hook(lambda grad: grad * Q_Influence)
        errQ_cat.backward(retain_variables=True)

        hook = errQ_cat.register_hook(lambda grad: grad * Q_Influence)
        errQ_con.backward()
#        errQ = errQ_cat + errQ_con
 #       hook = errQ.register_hook(lambda grad: grad * Q_Influence)
  #      errQ.backward(retain_variables=True)

        D_G_z2 = outputD.data.mean()
        optimizerQ.step()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q_cat: %.4f Loss_Q_con: %.4f     D(x): %.4f D(G(z)): %.4f | %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0], errG.data[0], errQ_cat.data[0], errQ_con.data[0], D_x, D_G_z1, D_G_z2))

        if i == len(dataloader)-1:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % options.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (options.outf, epoch),
                    normalize=True)


    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (options.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))
    torch.save(netQ.state_dict(), '%s/netQ_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation