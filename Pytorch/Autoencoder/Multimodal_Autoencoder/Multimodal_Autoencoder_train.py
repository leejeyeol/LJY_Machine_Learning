import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import torchvision.utils as vutils
from torch.autograd import Variable

import Pytorch.Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_model as model
import Pytorch.Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_dataloader as dset

# import custom package
import LJY_utils
import LJY_visualize_tools

# =======================================================================================================================
# Options
# =======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='KITTI', help='what is dataset?')
parser.add_argument('--image_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img',
                    help='path to dataset')
parser.add_argument('--depth_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/depth',
                    help='path to dataset')
parser.add_argument('--semantic_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/semantic',
                    help='path to dataset')
parser.add_argument('--fold', type=int, default=None)
parser.add_argument('--fold_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/fold_10',
                    help='folds maked by Preprocessing/fold_divider')

parser.add_argument('--net', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--outf', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
parser.add_argument('--imageSize', type=int, default=[212, 64], help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')
'''
parser.add_argument('--net_ER', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/encoder_R_epoch_0.pth', help='Encoder Red')
parser.add_argument('--net_EG', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/encoder_G_epoch_0.pth', help='Encoder Green')
parser.add_argument('--net_EB', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/encoder_B_epoch_0.pth', help='Encoder Blue')
parser.add_argument('--net_ED', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/encoder_D_epoch_0.pth', help='Encoder Depth')
parser.add_argument('--net_DR', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/decoder_R_epoch_0.pth', help='Decoder Red')
parser.add_argument('--net_DG', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/decoder_G_epoch_0.pth', help='Decoder Green')
parser.add_argument('--net_DB', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/decoder_B_epoch_0.pth', help='Decoder Blue')
parser.add_argument('--net_DD', default='/home/leejeyeol/Git/LJY_Machine_Learning/Autoencoder/Multimodal_Autoencoder/pretrained_model/decoder_D_epoch_0.pth', help='Decoder Depth')


parser.add_argument('--net_ER', default='', help='Encoder Red')
parser.add_argument('--net_EG', default='', help='Encoder Green')
parser.add_argument('--net_EB', default='', help='Encoder Blue')
parser.add_argument('--net_ED', default='', help='Encoder Depth')
parser.add_argument('--net_DR', default='', help='Decoder Red')
parser.add_argument('--net_DG', default='', help='Decoder Green')
parser.add_argument('--net_DB', default='', help='Decoder Blue')
parser.add_argument('--net_DD', default='', help='Decoder Depth')
'''
parser.add_argument('--net_E_RGB_R', default='', help='Encoder RGB Right')
parser.add_argument('--net_E_RGB_L', default='', help='Encoder RGB Left')
parser.add_argument('--net_E_D_D', default='', help='Encoder Depth Disparity')
parser.add_argument('--net_E_D_L', default='', help='Encoder Depth LIDAR')

parser.add_argument('--net_D_RGB_R', default='', help='Decoder RGB Right')
parser.add_argument('--net_D_RGB_L', default='', help='Decoder RGB Left')
parser.add_argument('--net_D_D_D', default='', help='Decoder Depth Disparity')
parser.add_argument('--net_D_D_L', default='', help='Decoder Depth LIDAR')

# Artificial data corruption ===========================================================================================
def salt_and_pepper(img, prob):
    """salt and pepper noise for mnist"""
    rnd = np.random.rand(img.shape)
    noisy = img[:]
    noisy[rnd < prob/2] = 0.
    noisy[rnd > 1 - prob/2] = 1.
    return noisy
def black_squares(img):
    W, H = img.shape
    x = rand(0,W)
    y = rand(0,H)

# ======================================================================================================================
# custom options

options = parser.parse_args()
print(options)

if options.fold is None:
    outf = options.outf
else:
    outf = os.path.join(options.outf, "fold_%d" % options.fold)

# save directory make   ================================================================================================
try:
    os.makedirs(outf)

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
if options.fold is None:
    dataloader = torch.utils.data.DataLoader(
        dset.MMAE_Dataloader(options.image_dataroot, options.depth_dataroot, options.semantic_dataroot, transform),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
else:
    dataloader = torch.utils.data.DataLoader(
        dset.fold_MMAE_Dataloader(options.fold, options.fold_dataroot, transform, 'train'),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# normalize to -1~1
ngpu = int(options.ngpu)

# =======================================================================================================================
# Models
# =======================================================================================================================
encoder_RGB_R = model.encoder(3)
encoder_RGB_R.apply(LJY_utils.weights_init)
if options.net_E_RGB_R != '':
    encoder_RGB_R.load_state_dict(torch.load(options.net_E_RGB_R))
print(encoder_RGB_R)

encoder_RGB_L = model.encoder(3)
encoder_RGB_L.apply(LJY_utils.weights_init)
if options.net_E_RGB_L != '':
    encoder_RGB_L.load_state_dict(torch.load(options.net_E_RGB_L))
print(encoder_RGB_L)

encoder_D_D = model.encoder(1)
encoder_D_D.apply(LJY_utils.weights_init)
if options.net_E_D_D != '':
    encoder_D_D.load_state_dict(torch.load(options.net_E_D_D))
print(encoder_D_D)

encoder_D_L = model.encoder(1)
encoder_D_L.apply(LJY_utils.weights_init)
if options.net_E_D_L != '':
    encoder_D_L.load_state_dict(torch.load(options.net_E_D_L))
print(encoder_D_L)

decoder_RGB_R = model.decoder(3)
decoder_RGB_R.apply(LJY_utils.weights_init)
if options.net_D_RGB_R != '':
    decoder_RGB_R.load_state_dict(torch.load(options.net_D_RGB_R ))
print(decoder_RGB_R)

decoder_RGB_L = model.decoder(3)
decoder_RGB_L.apply(LJY_utils.weights_init)
if options.net_D_RGB_L != '':
    decoder_RGB_L.load_state_dict(torch.load(options.net_D_RGB_L))
print(decoder_RGB_L)

decoder_D_D = model.decoder(1)
decoder_D_D.apply(LJY_utils.weights_init)
if options.net_D_D_D != '':
    decoder_D_D.load_state_dict(torch.load(options.net_D_D_D))
print(decoder_D_D)

decoder_D_L = model.decoder(1)
decoder_D_L.apply(LJY_utils.weights_init)
if options.net_D_D_L != '':
    decoder_D_L.load_state_dict(torch.load(options.net_D_D_L))
print(decoder_D_L)


# =======================================================================================================================
# Training
# =======================================================================================================================

def noise_mask(shape):
    mask = torch.Tensor(shape).fill_(0.9)
    mask = torch.bernoulli(mask)
    return mask


# criterion set
class RMSEloss(nn.Module):
    def forward(self, input, targets, mask, size_avarage=False):
        masked_input = torch.masked_select(input, mask)
        masked_targets = torch.masked_select(targets, mask)
        # return torch.sqrt(torch.mean((input - targets).pow(2))/targets.size()[1])
        return torch.sqrt(torch.mean((masked_input - masked_targets).pow(2)))


criterion = nn.MSELoss()
criterion_BCE = nn.BCELoss()
criterion_D = RMSEloss()


# setup optimizer   ====================================================================================================
optimizer = optim.Adam(list(encoder_RGB_R.parameters()) + list(decoder_RGB_R.parameters()) +
                       list(encoder_RGB_L.parameters()) + list(decoder_RGB_L.parameters()) +
                       list(encoder_D_D.parameters()) + list(decoder_D_D.parameters()) +
                       list(encoder_D_L.parameters()) + list(decoder_D_L.parameters()), betas=(0.5, 0.999), lr=2e-4)

# container generate
input_RGB_left = torch.FloatTensor(options.batchSize, 3, options.imageSize[0] * options.imageSize[1])
input_RGB_right = torch.FloatTensor(options.batchSize, 3, options.imageSize[0] * options.imageSize[1])
input_D_LIDAR = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])
input_D_disparity = torch.FloatTensor(options.batchSize, 1, options.imageSize[0] * options.imageSize[1])

if options.cuda:
    encoder_RGB_R.cuda()
    decoder_RGB_R.cuda()
    encoder_RGB_L.cuda()
    decoder_RGB_L.cuda()
    encoder_D_D.cuda()
    decoder_D_D.cuda()
    encoder_D_L.cuda()
    decoder_D_L.cuda()
    criterion.cuda()
    input_RGB_left = input_RGB_left.cuda()
    input_RGB_right = input_RGB_right.cuda()
    input_D_LIDAR = input_D_LIDAR.cuda()
    input_D_disparity = input_D_disparity.cuda()

# make to variables ====================================================================================================
input_RGB_left = Variable(input_RGB_left)
input_RGB_right = Variable(input_RGB_right)
input_D_LIDAR = Variable(input_D_LIDAR)
input_D_disparity = Variable(input_D_disparity)

win_dict = LJY_visualize_tools.win_dict()
line_win_dict = LJY_visualize_tools.win_dict()

# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (RGB_Left, RGB_Right, Depth_LIDAR, Depth_Disparity) in enumerate(dataloader, 0):

        original_RGB_Left = RGB_Left
        original_RGB_Left = Variable(original_RGB_Left).cuda()
        original_RGB_Right = RGB_Right
        original_RGB_Right = Variable(original_RGB_Right).cuda()

        original_Depth_LIDAR = Depth_LIDAR
        original_Depth_LIDAR = Variable(original_Depth_LIDAR).cuda()
        original_Depth_Disparity = Depth_Disparity
        original_Depth_Disparity = Variable(original_Depth_Disparity).cuda()


        '''
            if random.random() < 1 / 3:
            R = torch.zeros(R.shape)
            G = torch.zeros(G.shape)
            B = torch.zeros(B.shape)
            D = torch.zeros(D.shape)
        '''
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        optimizer.zero_grad()

        batch_size = RGB_Right.size(0)
        input_RGB_left.data.resize_(RGB_Left.size()).copy_(RGB_Left)
        input_RGB_right.data.resize_(RGB_Right.size()).copy_(RGB_Right)
        input_D_LIDAR.data.resize_(Depth_LIDAR.size()).copy_(Depth_LIDAR)
        input_D_disparity.data.resize_(Depth_Disparity.size()).copy_(Depth_Disparity)

        input_RGB_left_noise = torch.mul(RGB_Left, noise_mask(RGB_Left.shape))
        input_RGB_right_noise = torch.mul(RGB_Right, noise_mask(RGB_Right.shape))
        input_D_LIDAR_noise = torch.mul(Depth_LIDAR, noise_mask(Depth_LIDAR.shape))
        input_D_disparity_noise = torch.mul(Depth_Disparity, noise_mask(Depth_Disparity.shape))

        input_RGB_left_noise = Variable(input_RGB_left_noise).cuda()
        input_RGB_right_noise = Variable(input_RGB_right_noise).cuda()
        input_D_LIDAR_noise = Variable(input_D_LIDAR_noise).cuda()
        input_D_disparity_noise = Variable(input_D_disparity_noise).cuda()
        z_RGB_L = encoder_RGB_L(input_RGB_left_noise)
        z_RGB_R = encoder_RGB_R(input_RGB_right_noise)
        z_D_L = encoder_D_L(input_D_LIDAR_noise)
        z_D_D = encoder_D_D(input_D_disparity_noise)

        z = torch.cat((z_RGB_L, z_RGB_R, z_D_L, z_D_D), 1)

        output_RGB_L = decoder_RGB_L(z)
        output_RGB_R = decoder_RGB_R(z)
        output_D_L = decoder_D_L(z)
        output_D_D = decoder_D_D(z)


        err_RGB_L = criterion(output_RGB_L, original_RGB_Left)
        err_RGB_L.backward(retain_graph=True)

        err_RGB_R = criterion(output_RGB_R, original_RGB_Right)
        err_RGB_R.backward(retain_graph=True)

        err_D_L = criterion(output_D_L, original_Depth_LIDAR)
        err_D_L.backward(retain_graph=True)

        err_D_D = criterion(output_D_D, original_Depth_Disparity)
        err_D_D.backward(retain_graph=True)

        optimizer.step()

        # visualize
        print('[%d/%d][%d/%d] Loss_R: %f Loss_G: %f Loss_B: %f Loss_D: %f '
              % (epoch, options.iteration, i, len(dataloader),
                 err_RGB_L.data[0], err_RGB_R.data[0], err_D_L.data[0], err_D_D.data[0]))

        if True:
            RGB_L_Image = torch.cat((unorm(original_RGB_Left[0].data), unorm(output_RGB_L[0].data)), 1)
            RGB_R_Image = torch.cat((unorm(original_RGB_Right[0].data), unorm(output_RGB_R[0].data)), 1)
            Depth_L_Image = torch.cat((unorm(original_Depth_LIDAR[0].data), unorm(output_D_L[0].data)), 1)
            Depth_D_Image = torch.cat((unorm(original_Depth_Disparity[0].data), unorm(output_D_D[0].data)),1)

            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [RGB_L_Image, RGB_R_Image,
                                                                             Depth_L_Image, Depth_D_Image],
                                                                  ["RGB_L", "RGB_R", "Depth_L", "Depth_D"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_RGB_L.data.mean(), err_RGB_R.data.mean(),
                                                                       err_D_L.data.mean(), err_D_D.data.mean()],
                                                                      ['lossRGB_L', 'lossRGB_R', 'lossD_L', 'lossD_D'],
                                                                      epoch, i, len(dataloader))

    if True:
        torch.save(encoder_RGB_R.state_dict(), '%s/encoder_RGB_R_epoch_%d.pth' % (outf, epoch))
        torch.save(encoder_RGB_L.state_dict(), '%s/encoder_RGB_L_epoch_%d.pth' % (outf, epoch))
        torch.save(encoder_D_D.state_dict(), '%s/encoder_D_D_epoch_%d.pth' % (outf, epoch))
        torch.save(encoder_D_L.state_dict(), '%s/encoder_D_L_epoch_%d.pth' % (outf, epoch))
        torch.save(decoder_RGB_R.state_dict(), '%s/decoder_RGB_R_epoch_%d.pth' % (outf, epoch))
        torch.save(decoder_RGB_L.state_dict(), '%s/decoder_RGB_L_epoch_%d.pth' % (outf, epoch))
        torch.save(decoder_D_D.state_dict(), '%s/decoder_D_D_epoch_%d.pth' % (outf, epoch))
        torch.save(decoder_D_L.state_dict(), '%s/decoder_D_L_epoch_%d.pth' % (outf, epoch))

# Je Yeol. Lee \[T]/
# Jolly Co-operation