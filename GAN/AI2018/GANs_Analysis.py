import argparse
import os
import random
import glob as glob
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable



import GAN.AI2018.DCGAN.DCGAN_model as DCmodel
import GAN.AI2018.WGAN.WGAN_model as Wmodel
import GAN.AI2018.LSGAN.LSGAN_model as LSmodel
import GAN.AI2018.EBGAN.EBGAN_model as EBmodel
# import custom package
import LJY_utils

import numpy as np

from PIL import Image
# version conflict
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        cur_file_paths = glob.glob(self.base_path + '/*.*')
        cur_file_paths.sort()
        self.file_paths = cur_file_paths

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png', help='path to dataset')

parser.add_argument('--DC_G_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_results/DCGAN', help='path to dataset')
parser.add_argument('--W_G_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_results/WGAN', help='path to dataset')
parser.add_argument('--LS_G_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_results/LSGAN', help='path to dataset')
parser.add_argument('--EB_G_dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_results/EBGAN', help='path to dataset')

parser.add_argument('--net_DC_D', default='/home/leejeyeol/Git/LJY_Machine_Learning/GAN/AI2018/DCGAN/output/netD_epoch_90.pth', help="path of DCGAN Discriminator networks.(to continue training)")
parser.add_argument('--net_W_D', default='/home/leejeyeol/Git/LJY_Machine_Learning/GAN/AI2018/WGAN/output/netD_epoch_60.pth', help="path of WGAN Discriminator networks.(to continue training)")
parser.add_argument('--net_LS_D', default='/home/leejeyeol/Git/LJY_Machine_Learning/GAN/AI2018/LSGAN/output/netD_epoch_130.pth', help="path of LSGAN Discriminator networks.(to continue training)")
parser.add_argument('--net_EB_D', default='/home/leejeyeol/Git/LJY_Machine_Learning/GAN/AI2018/EBGAN/output/netD_epoch_4.pth', help="path of EBGAN Discriminator networks.(to continue training)")

parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False.')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=50000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=100, help='dimension of noise.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

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


# ======================================================================================================================
# Data and Parameters
# ======================================================================================================================
display = options.display
dataset= options.dataset
batch_size = options.batchSize
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)
nc = int(options.nc)

# CelebA call and load   ===============================================================================================
transform = transforms.Compose([
    transforms.CenterCrop(150),
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


dataloader = torch.utils.data.DataLoader(Dataloader(options.dataroot, transform),
                                         batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

DCdataloader = torch.utils.data.DataLoader(Dataloader(options.DC_G_dataroot, transform),
                                         batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

Wdataloader = torch.utils.data.DataLoader(Dataloader(options.W_G_dataroot, transform),
                                         batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

EBdataloader = torch.utils.data.DataLoader(Dataloader(options.EB_G_dataroot, transform),
                                         batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

LSdataloader = torch.utils.data.DataLoader(Dataloader(options.LS_G_dataroot, transform),
                                         batch_size=options.batchSize, shuffle=True, num_workers=options.workers)


MSE = nn.MSELoss(reduce=False)
# ======================================================================================================================
# Models
# ======================================================================================================================

# Discriminator ========================================================================================================
net_DC_D = DCmodel.Discriminator(ngpu, ndf, nc)
net_DC_D.apply(LJY_utils.weights_init)
if options.net_DC_D != '':
    net_DC_D.load_state_dict(torch.load(options.net_DC_D))
print(net_DC_D)

# Discriminator ========================================================================================================
net_W_D = Wmodel.Discriminator(ngpu, ndf, nc)
net_W_D.apply(LJY_utils.weights_init)
if options.net_W_D != '':
    net_DC_D.load_state_dict(torch.load(options.net_W_D))
print(net_W_D)

# Discriminator ========================================================================================================
net_LS_D = LSmodel.Discriminator(ngpu, ndf, nc)
net_LS_D.apply(LJY_utils.weights_init)
if options.net_LS_D != '':
    net_LS_D.load_state_dict(torch.load(options.net_LS_D))
print(net_LS_D)

# Discriminator ========================================================================================================
net_EB_D = EBmodel.Discriminator(ngpu, ndf, nc)
net_EB_D.apply(LJY_utils.weights_init)
if options.net_EB_D != '':
    net_EB_D.load_state_dict(torch.load(options.net_EB_D))
print(net_EB_D)


# container generate
noise = torch.FloatTensor(batch_size, nz, 1, 1)

if options.cuda:
    net_DC_D = net_DC_D.cuda()
    net_EB_D = net_EB_D.cuda()
    net_W_D = net_W_D.cuda()
    net_LS_D = net_LS_D.cuda()
    noise = noise.cuda()
    MSE = MSE.cuda()

# make to variables ====================================================================================================
noise = Variable(noise)


print("Test Real!")
data_iter = iter(dataloader)
DC_data_iter = iter(DCdataloader)
W_data_iter = iter(Wdataloader)
LS_data_iter = iter(LSdataloader)
EB_data_iter = iter(EBdataloader)


output_array = torch.from_numpy(np.zeros(shape=(50000, 4, 5), dtype=float))
# (number, D type(0:DCGAN, 1:WGAN, 2:LSGAN, 3:EBGAN), data generated by (0:DCGAN, 1:WGAN, 2:LSGAN, 3:EBGAN, 4:real)


i = 0
while i < len(dataloader):
    data = DC_data_iter.next()
    input = Variable(data)
    if options.cuda:
        input = input.cuda()
    output_DC_DC = net_DC_D(input)
    output_W_DC = net_W_D(input)
    output_LS_DC = net_LS_D(input)
    output_EB_DC = net_EB_D(input)
    output_EB_DC = MSE(output_EB_DC,input)

    data = W_data_iter.next()
    input = Variable(data)
    if options.cuda:
        input = input.cuda()
    output_DC_W = net_DC_D(input)
    output_W_W = net_W_D(input)
    output_LS_W = net_LS_D(input)
    output_EB_W = net_EB_D(input)
    output_EB_W = MSE(output_EB_W,input)

    data = LS_data_iter.next()
    input = Variable(data)
    if options.cuda:
        input = input.cuda()
    output_DC_LS = net_DC_D(input)
    output_W_LS = net_W_D(input)
    output_LS_LS = net_LS_D(input)
    output_EB_LS = net_EB_D(input)
    output_EB_LS = MSE(output_EB_LS,input)

    data = EB_data_iter.next()
    input = Variable(data)
    if options.cuda:
        input = input.cuda()
    output_DC_EB = net_DC_D(input)
    output_W_EB = net_W_D(input)
    output_LS_EB = net_LS_D(input)
    output_EB_EB = net_EB_D(input)
    output_EB_EB = MSE(output_EB_EB,input)

    data = data_iter.next()
    input = Variable(data)
    if options.cuda:
        input = input.cuda()
    output_DC_real = net_DC_D(input)
    output_W_real = net_W_D(input)
    output_LS_real = net_LS_D(input)
    output_EB_real = net_EB_D(input)
    output_EB_real = MSE(output_EB_real,input)

    for j in range(batch_size):
        output_array[i*batch_size+j][0][0] = output_DC_DC.data[j]
        output_array[i*batch_size+j][1][0] = output_W_DC.data[j]
        output_array[i*batch_size+j][2][0] = output_LS_DC.data[j]
        output_array[i*batch_size+j][3][0] = output_EB_DC.data[j].mean()

        output_array[i*batch_size+j][0][1] = output_DC_W.data[j]
        output_array[i*batch_size+j][1][1] = output_W_W.data[j]
        output_array[i*batch_size+j][2][1] = output_LS_W.data[j]
        output_array[i*batch_size+j][3][1] = output_EB_W.data[j].mean()

        output_array[i*batch_size+j][0][2] = output_DC_LS.data[j]
        output_array[i*batch_size+j][1][2] = output_W_LS.data[j]
        output_array[i*batch_size+j][2][2] = output_LS_LS.data[j]
        output_array[i*batch_size+j][3][2] = output_EB_LS.data[j].mean()

        output_array[i*batch_size+j][0][3] = output_DC_EB.data[j]
        output_array[i*batch_size+j][1][3] = output_W_EB.data[j]
        output_array[i*batch_size+j][2][3] = output_LS_EB.data[j]
        output_array[i*batch_size+j][3][3] = output_EB_EB.data[j].mean()

        output_array[i*batch_size+j][0][4] = output_DC_real.data[j]
        output_array[i*batch_size+j][1][4] = output_W_real.data[j]
        output_array[i*batch_size+j][2][4] = output_LS_real.data[j]
        output_array[i*batch_size+j][3][4] = output_EB_real.data[j].mean()

        print(output_array[i*batch_size+j])
    #visualize
    print('[%d/%d]' % (i, options.iteration/batch_size))
    i += 1

print(output_array.numpy().mean(axis=0))
print(output_array.numpy().std(axis=0))

np.save('%s/analysis.npy' % (options.outf),output_array.numpy())







# Je Yeol. Lee \[T]/