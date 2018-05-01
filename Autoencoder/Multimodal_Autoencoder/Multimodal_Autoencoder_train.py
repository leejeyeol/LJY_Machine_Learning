import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils
from torch.autograd import Variable

import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_model as model
import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_dataloader as dset

# import custom package
import LJY_utils
import LJY_visualize_tools

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='KITTI', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/train', help='path to dataset')
parser.add_argument('--net', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")


parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=[60, 18], help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options

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
    dset.MMAE_Dataloader(options.dataroot),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

# normalize to -1~1
ngpu = int(options.ngpu)


#=======================================================================================================================
# Models
#=======================================================================================================================
encoder_R = model.encoder()
encoder_R.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_R.load_state_dict(torch.load(options.netG))
print(encoder_R)

encoder_G = model.encoder()
encoder_G.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_G.load_state_dict(torch.load(options.netG))
print(encoder_G)

encoder_B = model.encoder()
encoder_B.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_B.load_state_dict(torch.load(options.netG))
print(encoder_B)

encoder_D = model.encoder()
encoder_D.apply(LJY_utils.weights_init)
if options.net != '':
    encoder_D.load_state_dict(torch.load(options.netG))
print(encoder_D)

decoder_R = model.decoder()
decoder_R.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_R.load_state_dict(torch.load(options.netD))
print(decoder_R)

decoder_G = model.decoder()
decoder_G.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_G.load_state_dict(torch.load(options.netD))
print(decoder_G)

decoder_B = model.decoder()
decoder_B.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_B.load_state_dict(torch.load(options.netD))
print(decoder_B)

decoder_D = model.decoder("d")
decoder_D.apply(LJY_utils.weights_init)
if options.net != '':
    decoder_D.load_state_dict(torch.load(options.netD))
print(decoder_D)
#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
class RMSEloss(nn.Module):
    def forward(self, input, targets, size_avarage=False):
        #return torch.sqrt(torch.mean((input - targets).pow(2))/targets.size()[1])
        return torch.sqrt(torch.mean((input - targets).pow(2)))

criterion = RMSEloss()



# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizer = optim.Adam(list(encoder_R.parameters()) + list(decoder_R.parameters())+
                       list(encoder_G.parameters()) + list(decoder_G.parameters())+
                       list(encoder_B.parameters()) + list(decoder_B.parameters())+
                       list(encoder_D.parameters()) + list(decoder_D.parameters()), betas=(0.5, 0.999), lr=1e-2)

# container generate
input_R = torch.FloatTensor(options.batchSize, 1, options.imageSize[0]*options.imageSize[1])
input_G = torch.FloatTensor(options.batchSize, 1, options.imageSize[0]*options.imageSize[1])
input_B = torch.FloatTensor(options.batchSize, 1, options.imageSize[0]*options.imageSize[1])
input_D = torch.FloatTensor(options.batchSize, 1, options.imageSize[0]*options.imageSize[1])





if options.cuda:
    encoder_R.cuda()
    decoder_R.cuda()
    encoder_G.cuda()
    decoder_G.cuda()
    encoder_B.cuda()
    decoder_B.cuda()
    encoder_D.cuda()
    decoder_D.cuda()
    criterion.cuda()
    input_R = input_R.cuda()
    input_G = input_G.cuda()
    input_B = input_B.cuda()
    input_D = input_D.cuda()
   


# make to variables ====================================================================================================
input_R = Variable(input_R)
input_G = Variable(input_G)
input_B = Variable(input_B)
input_D = Variable(input_D)

win_dict = LJY_visualize_tools.win_dict()

# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (R, G, B, D) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        optimizer.zero_grad()

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

        err_R = criterion(output_R, input_R)
        err_R.backward(retain_graph=True)

        err_G = criterion(output_G, input_G)
        err_G.backward(retain_graph=True)

        err_B = criterion(output_B, input_B)
        err_B.backward(retain_graph=True)

        err_D = criterion(output_D, input_D)
        err_D.backward(retain_graph=True)

        optimizer.step()

        #visualize
        print('[%d/%d][%d/%d] Loss_R: %f Loss_G: %f Loss_B: %f Loss_D: %f '
              % (epoch, options.iteration, i, len(dataloader),
                 err_R.data[0], err_G.data[0], err_B.data[0], err_D.data[0]))



        if options.display:
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict,
                                                            [torch.cat((R[0],G[0],B[0]),0).view(3,18,60),
                                                             torch.cat((output_R[0],output_G[0],output_B[0]),0).view(3,18,60).data,
                                                             D[0].view(18, 60),
                                                             output_D[0].view(18, 60).data],
                                                             ["RGB","RGB_recon","D","D_recon"])
        '''
        if i == len(dataloader)-1:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % options.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (options.outf, epoch),
                    normalize=True)
        '''
    if epoch % 100 == 0:
        torch.save(encoder_R.state_dict(), '%s/encoder_R_epoch_%d.pth' % (options.outf, epoch))
        torch.save(encoder_G.state_dict(), '%s/encoder_G_epoch_%d.pth' % (options.outf, epoch))
        torch.save(encoder_B.state_dict(), '%s/encoder_B_epoch_%d.pth' % (options.outf, epoch))
        torch.save(encoder_D.state_dict(), '%s/encoder_D_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder_R.state_dict(), '%s/decoder_R_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder_G.state_dict(), '%s/decoder_G_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder_B.state_dict(), '%s/decoder_B_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder_D.state_dict(), '%s/decoder_D_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation