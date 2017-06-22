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
parser.add_argument('--outf', default='./InfoGAN_Test', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='InfoGAN', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=62, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
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
batchSize = options.batchSize
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
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

#=======================================================================================================================
# generating
#=======================================================================================================================

# container generate
input = torch.FloatTensor(batchSize, 3, options.imageSize, options.imageSize)
final_noise = torch.FloatTensor(batchSize, nz+nconC+ncatC, 1, 1)
noise = torch.FloatTensor(batchSize, nz, 1, 1)

noise_c1 = torch.FloatTensor(batchSize, 1, 1, 1)
noise_c2 = torch.FloatTensor(batchSize, 1, 1, 1)
onehot_c = torch.FloatTensor(batchSize, 10)

# for check   ==========================================================================================================
fixed_noise = torch.FloatTensor(batchSize, nz+nconC+ncatC, 1, 1).normal_(0, 1)

label = torch.FloatTensor(batchSize)

if options.cuda:
    netG.cuda()
    input, label = input.cuda(), label.cuda()
    final_noise, noise, noise_c1, noise_c2, onehot_c = final_noise.cuda(), noise.cuda(), \
                                                                    noise_c1.cuda(),\
                                                                    noise_c2.cuda(), onehot_c.cuda()

# make to variables ====================================================================================================
input = Variable(input)
final_noise = Variable(final_noise)

noise = Variable(noise)
noise_c1 = Variable(noise_c1)
noise_c2 = Variable(noise_c2)
onehot_c = Variable(onehot_c)

# training start
print("Generating Start!")


# train with real data  ========================================================================================
# generate noise    ============================================================================================

c1 = False
c2 = True

noise.data.resize_(batchSize, nz, 1, 1)
noise.data.normal_(0, 1)
#nn.init.uniform(noise, 0.5, 0.5)


if c1:
    noise_c1.data.resize_(batchSize, 1, 1, 1)
    nn.init.uniform(noise_c1, -1, 1)
else:
    noise_c1.data = torch.FloatTensor((-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                       ))
    noise_c1.data.resize_(batchSize, 1, 1, 1)
    noise_c1=noise_c1.cuda()


if c2:
    noise_c2.data.resize_(batchSize, 1, 1, 1)
    nn.init.uniform(noise_c2, -1, 1)
else:
    noise_c2.data = torch.FloatTensor((-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                        -1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,
                                       ))
    noise_c2.data.resize_(batchSize, 1, 1, 1)
    noise_c2=noise_c2.cuda()


onehot_c.data = LJY_utils.one_hot((batchSize, 10), torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0,
                                                                     1, 1, 1, 1, 1, 1, 1, 1,
                                                                     2, 2, 2, 2, 2, 2, 2, 2,
                                                                     3, 3, 3, 3, 3, 3, 3, 3,
                                                                     4, 4, 4, 4, 4, 4, 4, 4,
                                                                     5, 5, 5, 5, 5, 5, 5, 5,
                                                                     6, 6, 6, 6, 6, 6, 6, 6,
                                                                     7, 7, 7, 7, 7, 7, 7, 7,
                                                                     8, 8, 8, 8, 8, 8, 8, 8,
                                                                     9, 9, 9, 9, 9, 9, 9, 9]).view(-1,1)).cuda()

onehot_c = onehot_c.float()
onehot_c.data.resize_(batchSize, 10, 1, 1)

final_noise = torch.cat((noise, noise_c1, noise_c2, onehot_c), 1)
fake = netG(final_noise)
vutils.save_image(fake.data,
        '%s/generate_samples.png' % (options.outf),
        normalize=False)



# Je Yeol. Lee \[T]/
# Jolly Co-operation