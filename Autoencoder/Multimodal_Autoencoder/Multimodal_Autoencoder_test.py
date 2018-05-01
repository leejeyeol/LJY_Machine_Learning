import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from visdom import Visdom

import torchvision.utils as vutils
import torchvision
from torch.autograd import Variable

import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_model as model
import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_dataloader as dset

# import custom package
import LJY_utils

# =======================================================================================================================
# Options
# =======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='KITTI', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/val',
                    help='path to dataset')
parser.add_argument('--net', default='./pretrained_model', help="path of pretrained model")
parser.add_argument('--pretrained_epoch', default=100, help="epoch of pretrained model")

parser.add_argument('--outf', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/KITTI/RGB', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=[60, 18], help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')

# custom options

options = parser.parse_args()
print(options)

if options.display:
    vis = Visdom()

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

# =======================================================================================================================
# Data and Parameters
# =======================================================================================================================

# MNIST call and load   ================================================================================================

dataloader = torch.utils.data.DataLoader(
    dset.MMAE_Dataloader(options.dataroot),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

# normalize to -1~1
ngpu = int(options.ngpu)

# =======================================================================================================================
# Models
# =======================================================================================================================
encoder_R = model.encoder()
encoder_R.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_R.load_state_dict(torch.load(os.path.join(options.net, "encoder_R_epoch_%d.pth"%options.pretrained_epoch)))
print(encoder_R)

encoder_G = model.encoder()
encoder_G.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_G.load_state_dict(torch.load(os.path.join(options.net, "encoder_G_epoch_%d.pth"%options.pretrained_epoch)))
print(encoder_G)

encoder_B = model.encoder()
encoder_B.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_B.load_state_dict(torch.load(os.path.join(options.net, "encoder_B_epoch_%d.pth"%options.pretrained_epoch)))
print(encoder_B)

encoder_D = model.encoder()
encoder_D.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_D.load_state_dict(torch.load(os.path.join(options.net, "encoder_D_epoch_%d.pth"%options.pretrained_epoch)))
print(encoder_D)

decoder_R = model.decoder()
decoder_R.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_R.load_state_dict(torch.load(os.path.join(options.net, "decoder_R_epoch_%d.pth"%options.pretrained_epoch)))
print(decoder_R)

decoder_G = model.decoder()
decoder_G.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_G.load_state_dict(torch.load(os.path.join(options.net, "decoder_G_epoch_%d.pth"%options.pretrained_epoch)))
print(decoder_G)

decoder_B = model.decoder()
decoder_B.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_B.load_state_dict(torch.load(os.path.join(options.net, "decoder_B_epoch_%d.pth"%options.pretrained_epoch)))
print(decoder_B)

decoder_D = model.decoder()
decoder_D.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_D.load_state_dict(torch.load(os.path.join(options.net, "decoder_D_epoch_%d.pth"%options.pretrained_epoch)))
print(decoder_D)


# =======================================================================================================================
# Training
# =======================================================================================================================
input_R = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])
input_G = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])
input_B = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])
input_D = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])

if options.cuda:
    encoder_R.cuda()
    decoder_R.cuda()
    encoder_G.cuda()
    decoder_G.cuda()
    encoder_B.cuda()
    decoder_B.cuda()
    encoder_D.cuda()
    decoder_D.cuda()
    input_R = input_R.cuda()
    input_G = input_G.cuda()
    input_B = input_B.cuda()
    input_D = input_D.cuda()

# make to variables ====================================================================================================
input_R = Variable(input_R)
input_G = Variable(input_G)
input_B = Variable(input_B)
input_D = Variable(input_D)

# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (R, G, B, D) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================


        batch_size = R.size(0)
        input_R.data.resize_(R.size()).copy_(R)
        input_G.data.resize_(G.size()).copy_(G)
        input_B.data.resize_(B.size()).copy_(B)
        input_D.data.resize_(D.size()).copy_(D)

        z_R = encoder_R(input_R)
        z_G = encoder_G(input_G)
        z_B = encoder_B(input_B)
        z_D = encoder_D(input_D)

        z = torch.cat((z_R, z_G, z_B, z_D), 1)

        output_R = decoder_R(z)
        output_G = decoder_G(z)
        output_B = decoder_B(z)
        output_D = decoder_D(z)



        # visualize
        print('[%d/%d][%d/%d]'
              % (epoch, options.iteration, i, len(dataloader)))


        if os.path.basename(options.outf) == "RGB":
            vutils.save_image(torch.cat((R,G,B),0).view(3,18,60), '%s/real_samples_%04d.png' % (options.outf, i))
            vutils.save_image(torch.cat((output_R,output_G,output_B),0).view(3,18,60).data, '%s/reconstruction_%04d.png' % (options.outf, i))
        if os.path.basename(options.outf) == "Depth":
            vutils.save_image(D.view(18, 60), '%s/real_samples_%04d.png' % (options.outf, i), normalize=True)
            vutils.save_image(output_D.view(18, 60).data,'%s/reconstruction_%04d.png' % (options.outf, i), normalize=True)

        print("debug")



        # Je Yeol. Lee \[T]/
        # Jolly Co-operation