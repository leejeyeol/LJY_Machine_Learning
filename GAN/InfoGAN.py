import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret


#======================================================================================================================
# Options
#======================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='DCGAN', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=62, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")
parser.add_argument('--outf', default='.', help="folder to output images and model checkpoints")
parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)


# save directory make
try:
    os.makedirs(options.outf)
except OSError:
    pass

# seed set
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


#======================================================================================================================
# Data and Parameters
#======================================================================================================================

# MNIST call and load
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)
nc = int(options.nc)
nconC = 2
ncatC = 10

#======================================================================================================================
# Models
#======================================================================================================================


# todo netG add c
# todo netD add Q()
# todo result visualize
# todo add 4 datasets and models


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



# Generator ============================================================================================================
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nz*1*1 => 1024*1*1
            nn.ConvTranspose2d(in_channels=nz+nconC+ncatC, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 1024*1*1 => 128*7*7
            nn.ConvTranspose2d(1024, 128, 7, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128*7*7 => 64*14,14
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64*14*14 => 1*28*28
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

netG = _netG(ngpu)
netG.apply(weights_init)
if options.netG != '':
    netG.load_state_dict(torch.load(options.netG))
print(netG)

# Discriminator ========================================================================================================
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main_netD = nn.Sequential(
            # 1024 => 1
            nn.Conv2d(1024, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.main = nn.Sequential(
            # 1*28*28 => 64*14*14
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64*14*14 => 128*7*7
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128*7*7 => 1024
            nn.Conv2d(128, 1024, 7, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            outputD = nn.parallel.data_parallel(self.main_netD, output, range(self.ngpu))
        else:
            output = self.main(input)
            outputD = self.main_netD(output)
        return outputD, output

netD = _netD(ngpu)
netD.apply(weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)

# Auxiliary distribution ===============================================================================================
class _netQ(nn.Module):
    def __init__(self, ngpu):
        super(_netQ, self).__init__()
        self.ngpu = ngpu
        self.main_netQ = nn.Sequential(
            # 1024 => 128
            nn.Conv2d(1024, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cat_netQ = nn.Sequential(
            # 128 -> onehot code
            nn.Conv2d(128, ncatC, 1, 1, 0, bias=False),
            nn.Softmax2d()
        )
        self.con_netQ = nn.Sequential(
            # 128 ->
            nn.Conv2d(128, nconC, 1, 1, 0, bias=False),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputQ = nn.parallel.data_parallel(self.main_netQ, input, range(self.ngpu))
            c_cont = nn.parallel.data_parallel(self.con_netQ, outputQ, range(self.ngpu))
            c_cat = nn.parallel.data_parallel(self.cat_netQ, outputQ, range(self.ngpu))
        else:
            outputQ = self.main_netQ(input)
            c_cont = self.con_netQ(outputQ)
            c_cat = self.cat_netQ(outputQ)
        return c_cat, c_cont

netQ = _netQ(ngpu)
netQ.apply(weights_init)
if options.netQ != '':
    netQ.load_state_dict(torch.load(options.netQ))
print(netQ)



#======================================================================================================================
# Training
#======================================================================================================================

criterion = nn.BCELoss()


input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)
final_noise = torch.FloatTensor(options.batchSize, nz+nconC+ncatC, 1, 1)
noise = torch.FloatTensor(options.batchSize, nz, 1, 1)

noise_c1 = torch.FloatTensor(options.batchSize, 1, 1, 1)
noise_c2 = torch.FloatTensor(options.batchSize, 1, 1, 1)
onehot_c = torch.FloatTensor(options.batchSize, 10)

#for test
fixed_noise = torch.FloatTensor(options.batchSize, nz+nconC+ncatC, 1, 1).normal_(0, 1)

label = torch.FloatTensor(options.batchSize)
real_label = 1
fake_label = 0

if options.cuda:
    netD.cuda()
    netG.cuda()
    netQ.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    final_noise, noise, fixed_noise, noise_c1, noise_c2, onehot_c = final_noise.cuda(), noise.cuda(), fixed_noise.cuda(), noise_c1.cuda(),\
                                                         noise_c2.cuda(), onehot_c.cuda()

# make to variables
input = Variable(input)
label = Variable(label)

final_noise = Variable(final_noise)

noise = Variable(noise)
noise_c1 = Variable(noise_c1)
noise_c2 = Variable(noise_c2)
onehot_c = Variable(onehot_c)

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
optimizerQ = optim.Adam(netQ.parameters(), lr=options.lr, betas=(options.beta1, 0.999))


for epoch in range(options.iteration):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real data
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        outputD, midQ = netD(input)
        errD_real = criterion(outputD, label)
        errD_real.backward()
        D_x = outputD.data.mean()

        # generate noise
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        noise_c1.data.resize_(batch_size, 1, 1, 1)
        noise_c1.data.normal_(0, 1)
        noise_c2.data.resize_(batch_size, 1, 1, 1)
        noise_c2.data.normal_(0, 1)

        onehot_c.data = one_hot((batch_size, 10), torch.LongTensor([random.randrange(0, 10) for i in range(batch_size)]).view(-1, 1)).cuda()
        onehot_c = onehot_c.float()
        onehot_c.data.resize_(batch_size, 10, 1, 1)

        final_noise = torch.cat((noise, noise_c1, noise_c2, onehot_c), 1)

        #train with fake data
        fake = netG(final_noise)
        label.data.fill_(fake_label)

        outputD, midQ = netD(fake.detach())
        c_cat, c_cont = netQ(midQ.detach())

        errD_fake = criterion(outputD, label)
        errD_fake.backward()
        errQ_fake = criterion(torch.cat((c_cat, c_cont), 1), torch.cat((onehot_c, noise_c1, noise_c2), 1))
        errQ_fake.backward()

        D_G_z1 = outputD.data.mean()
        errD = errD_real + errD_fake

        optimizerD.step()
        optimizerQ.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        outputD, midQ = netD(fake)
        errG = criterion(outputD, label)
        errG.backward()
        D_G_z2 = outputD.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 937 == 0:
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

# Je Yeol Lee
# \[T]/
# Jolly Co-operation