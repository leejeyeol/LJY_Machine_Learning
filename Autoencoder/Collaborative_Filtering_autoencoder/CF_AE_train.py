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
from torch.autograd import Function
import torch.nn.functional as F
import Autoencoder.Collaborative_Filtering_autoencoder.CF_AE_dataloader as dset
import Autoencoder.Collaborative_Filtering_autoencoder.CF_AE_model as model
# import custom package
import LJY_utils

# ======================================================================================================================
#  Options
# ======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='Movielens', help='what is dataset?')
parser.add_argument('--dataroot', default='/home/leejeyeol/Downloads/ml-20m/user_item_matrix_ml-20m.npy', help='path to dataset')
parser.add_argument('--net', default='', help="path of network")
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
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options

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

dataloader = torch.utils.data.DataLoader(
    dset.CF_AE_Dataloader(options.dataroot),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers, drop_last=False)
# normalize to -1~1
ngpu = int(options.ngpu)

class MMSEloss(nn.Module):

    def forward(self, input, targets, size_avarage=False):
        mask = targets != 0
        num_ratings = torch.sum(mask.float())
        criterion = nn.MSELoss(size_average=size_avarage)
        return criterion(input * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_avarage else num_ratings
'''
def MMSEloss(inputs, targets, size_avarage=False):
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(size_average=size_avarage)
    return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_avarage else num_ratings
'''

# ======================================================================================================================
# Models
# ======================================================================================================================

# Generator ============================================================================================================
encoder = model.encoder()
encoder.apply(LJY_utils.weights_init)
if options.net != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

# Discriminator ========================================================================================================
decoder = model.decoder()
decoder.apply(LJY_utils.weights_init)
if options.net != '':
    decoder.load_state_dict(torch.load(options.netD))
print(decoder)


#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
criterion = MMSEloss()
criterion2 = nn.MSELoss()

# setup optimizer   ====================================================================================================

optimizer_e = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=0.02)
optimizer_d = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=0.02)


# container generate
input = torch.FloatTensor(options.batchSize, 1011)

if options.cuda:
    encoder=encoder.cuda()
    decoder=decoder.cuda()
    criterion = criterion.cuda()
    criterion2 = criterion2.cuda()
    input = input.cuda()


# make to variables ====================================================================================================
input = Variable(input)


# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (data) in enumerate(dataloader, 0):

        data = data.cuda()

        optimizer_e.zero_grad()
        optimizer_d.zero_grad()


        batch_size = data.size(0)
        input.data.resize_(data.size()).copy_(data)

        z = encoder(input)
        output = decoder(z)

        err, bot = criterion(output, input)
        err = err/bot
        err.backward()
        optimizer_e.step()
        optimizer_d.step()


        # dense re-feeding
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        new_input = output.detach()

        z2 = encoder(new_input)
        output2 = decoder(z2)

        err2 = criterion2(output2, new_input)
        err2.backward()
        optimizer_e.step()
        optimizer_d.step()



        #visualize
        print('[%d/%d][%d/%d] Loss_1: %f Loss_refeeding: %f'
              % (epoch, options.iteration, i, len(dataloader),
                 err.data[0], err2.data[0]))

        #if i == len(dataloader)-1:

    # do checkpointing
    if IsSave == True:
        torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder.state_dict(), '%s/decoder_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation