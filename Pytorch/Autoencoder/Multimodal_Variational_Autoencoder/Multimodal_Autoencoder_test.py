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

import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_model as model
import Autoencoder.Multimodal_Autoencoder.Multimodal_Autoencoder_dataloader as dset

# import custom package
import LJY_utils
import LJY_visualize_tools

# =======================================================================================================================
# Options
# =======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='KITTI', help='what is dataset?')
parser.add_argument('--image_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img_ppm',
                    help='path to dataset')
parser.add_argument('--depth_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/depth',
                    help='path to dataset')
parser.add_argument('--semantic_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/semantic',
                    help='path to dataset')
parser.add_argument('--fold',type=int, default=None)
parser.add_argument('--fold_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/fold_10',
                    help='folds maked by Preprocessing/fold_divider')

parser.add_argument('--net', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=[608, 96], help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--net_ER', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/encoder_R_epoch_5.pth', help='Encoder Red')
parser.add_argument('--net_EG', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/encoder_G_epoch_5.pth', help='Encoder Green')
parser.add_argument('--net_EB', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/encoder_B_epoch_5.pth', help='Encoder Blue')
parser.add_argument('--net_ED', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/encoder_D_epoch_5.pth', help='Encoder Depth')
parser.add_argument('--net_DR', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/decoder_R_epoch_5.pth', help='Decoder Red')
parser.add_argument('--net_DG', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/decoder_G_epoch_5.pth', help='Decoder Green')
parser.add_argument('--net_DB', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/decoder_B_epoch_5.pth', help='Decoder Blue')
parser.add_argument('--net_DD', default='/media/leejeyeol/74B8D3C8B8D38750/experiment/MMAE/fold_0/decoder_D_epoch_5.pth', help='Decoder Depth')


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
        dset.fold_MMAE_Dataloader(options.fold, options.fold_dataroot, transform, 'validation'),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# normalize to -1~1
ngpu = int(options.ngpu)

# =======================================================================================================================
# Models
# =======================================================================================================================
encoder_R = model.encoder()
encoder_R.apply(LJY_utils.weights_init)
if options.net_ER != '':
    encoder_R.load_state_dict(torch.load(options.net_ER))
print(encoder_R)

encoder_G = model.encoder()
encoder_G.apply(LJY_utils.weights_init)
if options.net_EG != '':
    encoder_G.load_state_dict(torch.load(options.net_EG))
print(encoder_G)

encoder_B = model.encoder()
encoder_B.apply(LJY_utils.weights_init)
if options.net_EB != '':
    encoder_B.load_state_dict(torch.load(options.net_EB))
print(encoder_B)

encoder_D = model.encoder()
encoder_D.apply(LJY_utils.weights_init)
if options.net_ED != '':
    encoder_D.load_state_dict(torch.load(options.net_ED))
print(encoder_D)

decoder_R = model.decoder()
decoder_R.apply(LJY_utils.weights_init)
if options.net_DR != '':
    decoder_R.load_state_dict(torch.load(options.net_DR))
print(decoder_R)

decoder_G = model.decoder()
decoder_G.apply(LJY_utils.weights_init)
if options.net_DG != '':
    decoder_G.load_state_dict(torch.load(options.net_DG))
print(decoder_G)

decoder_B = model.decoder()
decoder_B.apply(LJY_utils.weights_init)
if options.net_DB != '':
    decoder_B.load_state_dict(torch.load(options.net_DB))
print(decoder_B)

decoder_D = model.decoder()
decoder_D.apply(LJY_utils.weights_init)
if options.net_DD != '':
    decoder_D.load_state_dict(torch.load(options.net_DD))
print(decoder_D)

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


# criterion = RMSEloss()
criterion = nn.MSELoss()
criterion_BCE = nn.BCELoss()
criterion_D = RMSEloss()

# setup optimizer   ====================================================================================================


# container generate
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
line_win_dict = LJY_visualize_tools.win_dict()
mean_err_R = []
mean_err_G = []
mean_err_B = []
mean_err_D = []
# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (R, G, B, D, D_mask) in enumerate(dataloader, 0):
        original_R = R
        original_G = G
        original_B = B
        original_D = D
        D_mask=Variable(D_mask).cuda()
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        R = torch.zeros(R.shape)
        G = torch.zeros(G.shape)
        B = torch.zeros(B.shape)
        #D = torch.zeros(D.shape)

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

        err_G = criterion(output_G, input_G)

        err_B = criterion(output_B, input_B)

        err_D = criterion_D(output_D, input_D, D_mask)


        # visualize
        print('[%d/%d][%d/%d] Loss_R: %f Loss_G: %f Loss_B: %f Loss_D: %f '
              % (epoch, options.iteration, i, len(dataloader),
                 err_R.data[0], err_G.data[0], err_B.data[0], err_D.data[0]))

        if True:

            RGB_Image = torch.cat((unorm(torch.cat((R[0], G[0], B[0]), 0)),
                                   unorm(torch.cat((output_R[0].data, output_G[0].data, output_B[0].data), 0)).cpu(),
                                   unorm(torch.cat((original_R[0], original_G[0], original_B[0]), 0))),
                                  1)
            Depth_Image = torch.cat((unorm(D[0]), unorm(output_D[0].data).cpu(), unorm(original_D[0])), 1)

            Test_Input_Image = torch.cat((unorm(R[0]), unorm(G[0]), unorm(B[0]), unorm(original_D[0])), 1)
            Test_Output_Image = torch.cat((unorm(output_R[0].data), unorm(output_G[0].data), unorm(output_B[0].data), unorm(output_D[0].data)), 1).cpu()

            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [RGB_Image, Depth_Image], ["RGB", "Depth"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_R.data.mean(), err_G.data.mean(),
                                                                       err_B.data.mean(), err_D.data.mean()],
                                                                      ['lossR', 'lossG', 'lossB', 'lossD'], epoch,
                                                                      i,
                                                                      len(dataloader))
            mean_err_R.append(err_R.data.mean())
            mean_err_G.append(err_G.data.mean())
            mean_err_B.append(err_B.data.mean())
            mean_err_D.append(err_D.data.mean())
print(np.array(mean_err_R).mean())
print(np.array(mean_err_G).mean())
print(np.array(mean_err_B).mean())
print(np.array(mean_err_D).mean())
# Je Yeol. Lee \[T]/
# Jolly Co-operation
