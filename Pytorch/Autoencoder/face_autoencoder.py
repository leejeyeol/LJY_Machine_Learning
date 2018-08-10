import torch.utils.data as ud
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import LJY_utils
import LJY_visualize_tools

import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import torch.utils.data
import numpy as np

class DL(torch.utils.data.Dataset):
    def __init__(self, path, transform,type ='train'):
        random.seed = 1
        super().__init__()
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()
        total_file_paths = []
        cur_file_paths = glob.glob(os.path.join(self.base_path ,'CMU_PIE', '*'))
        total_file_paths = total_file_paths+cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path ,'YaleB', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        random.shuffle(total_file_paths)
        num_of_valset = 200
        self.val_file_paths = sorted(total_file_paths[:num_of_valset])
        self.file_paths = sorted(total_file_paths[num_of_valset:])

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.resize((64,64))

    def __len__(self):
        if self.type == 'train':
            return len(self.file_paths)
        elif self.type == 'test':
            return len(self.val_file_paths)

    def __getitem__(self, item):
        if self.type == 'train':
            path = self.file_paths[item]
        elif self.type == 'test':
            path = self.val_file_paths[item]

        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


class encoder(nn.Module):
    def __init__(self,  z_size=2, channel=3, num_filters=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, z_size, 4, 2, 1, bias=False),
        )
        # init weights
        self.weight_init()

    def forward(self, x):
        #AE
        z = self.encoder(x)
        return z

    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder(nn.Module):
    def __init__(self, z_size=2,channel=3, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def make_random_squre(image):
    image_ = image
    x = random.randrange(0, 64)
    y = random.randrange(0, 64)
    w = random.randrange(5, 30)
    h = random.randrange(5, 50)
    if y+h < 63 and x+w < 63:
        image_[:, :, x:x + w, y:y + h] = -1
        return image_
    else:
        return make_random_squre(image_)


#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='celebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/img', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=10000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=300, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=500, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')

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


#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

ngpu = int(options.ngpu)
nz = int(options.nz)

encoder = encoder(options.nz, options.nc)
encoder.apply(LJY_utils.weights_init)
if options.netG != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

decoder = decoder(options.nz, options.nc)
decoder.apply(LJY_utils.weights_init)
if options.netD != '':
    decoder.load_state_dict(torch.load(options.netD))
print(decoder)

#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()

# setup optimizer   ====================================================================================================
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-3)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-3)


# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)

if options.cuda:
    encoder.cuda()
    decoder.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    input = input.cuda()


# make to variables ====================================================================================================
input = Variable(input)

# training start
def train():
    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    dataloader = torch.utils.data.DataLoader(
        DL(options.dataroot, transform, 'train'),
        batch_size=options.batchSize, shuffle=True)


    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")
    for epoch in range(options.iteration):
        for i, data in enumerate(dataloader, 0):
            # autoencoder training  ====================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            original_data = Variable(real_cpu).cuda()
            #input.data.resize_(real_cpu.size()).copy_(make_random_squre(real_cpu))
            input.data.resize_(real_cpu.size()).copy_(real_cpu)




            z = encoder(input)
            x_recon = decoder(z)

            err_mse = MSE_loss(x_recon, original_data.detach())
            err_mse.backward(retain_graph=True)

            optimizerE.step()
            optimizerD.step()
            #visualize
            print('[%d/%d][%d/%d] Loss: %.4f'% (epoch, options.iteration, i, len(dataloader), err_mse.data.mean()))
            testImage = torch.cat((unorm(original_data.data[0]),unorm(input.data[0]), unorm(x_recon.data[0])), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_mse.data.mean(),0],
                                                                      ['loss_recon_x','zero'],
                                                                      epoch, i, len(dataloader))

        # do checkpointing
        if epoch % 9 == 0:
            torch.save(encoder.state_dict(), '%s/face_6_encoder_epoch_%d.pth' % (options.outf, epoch+1))
            torch.save(decoder.state_dict(), '%s/face_6_decoder_epoch_%d.pth' % (options.outf, epoch+1))

def test():
    in_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()

    print("Testing Start!")
    trained = 76
    for j in range(trained):
        ep = (j)*9 + 1
        encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_6_encoder_epoch_%d.pth")%ep))
        decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_6_decoder_epoch_%d.pth")%ep))

        transform = transforms.Compose([
            transforms.Scale(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        dataloader = torch.utils.data.DataLoader(DL(options.dataroot, transform,'test'),
                                                 batch_size=200, shuffle=True)



        for i, data in enumerate(dataloader, 0):
            real_cpu = data

            original_data = Variable(real_cpu).cuda()
            z = encoder(original_data)
            x_recon = decoder(z)
            err_mse = MSE_loss(x_recon, original_data.detach())

            input_bbox = Variable(make_random_squre(real_cpu)).cuda()
            z = encoder(input_bbox)
            x_recon_bbox = decoder(z)
            err_mse2 = MSE_loss(x_recon_bbox, input_bbox.detach())
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_mse.data.mean(), err_mse2.data.mean(), 0],
                                                                      ['nobox', 'box', 'zero'],
                                                                      j, i, len(dataloader))

            #testImage = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(input_bbox.data[0]), unorm(x_recon_bbox.data[0])), 2)
            #toimg = transforms.ToPILImage()
            #toimg(testImage.cpu()).save("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result"+"/%05d.png"%i)
            #win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            print("[%d/%d]"%(ep,trained))
def test_img():
    win_dict = LJY_visualize_tools.win_dict()

    print("Testing Start!")


    ep = 775
    encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_6_encoder_epoch_%d.pth")%ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_6_decoder_epoch_%d.pth")%ep))

    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    dataloader = torch.utils.data.DataLoader(DL(options.dataroot, transform,'test'),
                                             batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader, 0):
        real_cpu = data

        original_data = Variable(real_cpu).cuda()
        z = encoder(original_data)
        x_recon = decoder(z)

        input_bbox = Variable(make_random_squre(real_cpu)).cuda()
        z = encoder(input_bbox)
        x_recon_bbox = decoder(z)


        testImage = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(input_bbox.data[0]), unorm(x_recon_bbox.data[0])), 2)
        toimg = transforms.ToPILImage()
        toimg(testImage.cpu()).save("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result"+"/%05d.png"%i)
        win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

        print("[%d/%d]"%(i,len(dataloader)))

if __name__ == "__main__" :
    #train()
    #test()
    test_img()



# Je Yeol. Lee \[T]/
# Jolly Co-operation