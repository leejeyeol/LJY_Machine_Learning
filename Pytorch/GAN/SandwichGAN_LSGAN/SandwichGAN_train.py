import argparse
import random
import torch.nn as nn
import torch

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

import os
import glob
import torch.utils.data
import numpy as np

from torch.autograd import Variable
# import custom package
import LJY_utils
import LJY_visualize_tools
#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='HMDB51', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/HMDB51/middle_block224x224', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./output(for_test)', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=200, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)
'''
class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

def criterion_reconstruction_perceptual(input,target):
    feature_input = loss_net(input)
    feature_target = loss_net(target)
    loss = 0
    for i in range(4):
        loss = loss + criterion_MSE(feature_input[i], feature_target[i])

    return loss/4
'''


win_dict = LJY_visualize_tools.win_dict()
line_win_dict = LJY_visualize_tools.win_dict()

def noise_mask(shape, prob = 0.9):
    mask = torch.Tensor(shape).fill_(prob)
    mask = torch.bernoulli(mask)
    return mask



class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class _encoder(nn.Module):
    def __init__(self, ngpu, nz=200):
        super(_encoder, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.encoder_residual_block = nn.Sequential(
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256),
            ResidualBlock(dim_in=256, dim_out=256))
        self.vectorize = nn.Conv2d(256, nz, 16)


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self._encoder, input, range(self.ngpu))
        else:
            output = self.encoder(input) # 80 64 64 64
            output = self.encoder_residual_block(output) # 80 256 16 16
            output = self.vectorize(output)
        return output

# Generator
class _netG(nn.Module):
    def __init__(self, ngpu, nz=200):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.netG= nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 2, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            outputG = nn.parallel.data_parallel(self.main_netG, input, range(self.ngpu))
        else:
            outputG = self.netG(input) # 80 128 32 32

        return outputG

# Discriminator
class _netD(nn.Module):
    def __init__(self, N=1000, nz=200):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(nz, N),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),

            nn.Linear(N, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        cls = self.discriminator(z)
        return cls



class SandwichGAN_Dataloader(torch.utils.data.Dataset):
    #todo
    def __init__(self, path, transform, centered=False):
        super().__init__()
        self.transform = transform
        self.centered = centered
        self.add_string = lambda a, b: a + b

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

        cur_file_paths = glob.glob(self.base_path + '/*.npy')
        cur_file_paths.sort()
        self.file_paths = cur_file_paths


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        data = np.load(self.file_paths[item])
        pre_frame, mid_frame, nxt_frame = data

        pre_frame = self.transform(pre_frame)
        mid_frame = self.transform(mid_frame)
        nxt_frame = self.transform(nxt_frame)

        return pre_frame, mid_frame, nxt_frame

    #todo
    def get_decenterd_data(self, centered_data):
        result = centered_data.mul_(255) + self.mean_image
        result = result.byte()
        return result
    #todo
    def get_mean_image(self):
        mean_image = np.load(os.path.join(os.path.dirname(self.base_path), "mean_image.npy"))
        mean_image = np.transpose(mean_image, (2, 0, 1))
        mean_image = torch.from_numpy(mean_image).float()
        return mean_image



print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!please remake dataset ! no normalize")

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
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

dataloader = torch.utils.data.DataLoader(
    SandwichGAN_Dataloader(options.dataroot,transform,centered=True),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers, drop_last=False)
# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)
ngf = int(options.ngf)
ndf = int(options.ndf)


#=======================================================================================================================
# Models
#=======================================================================================================================

# Generator ============================================================================================================
decoder = _netG(ngpu, nz=nz)
decoder.apply(LJY_utils.weights_init)
if options.netG != '':
    decoder.load_state_dict(torch.load(options.netG))
print(decoder)

encoder = _encoder(ngpu, nz=nz)
encoder.apply(LJY_utils.weights_init)
if options.netG != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

# Discriminator ========================================================================================================
netD = _netD(N=1000, nz=nz*3)
netD.apply(LJY_utils.weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)

#loss_net =VGG16()
#print(loss_net)


#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set

criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()
criterion_MSE = nn.MSELoss()

# setup optimizer   ====================================================================================================

#Q_Influence = 1.0
# todo add betas=(0.5, 0.999),
optimizerD_SGD = optim.SGD(netD.parameters(), lr=2e-4)
optimizerD_Adam = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerDecoder_SGD = optim.SGD(decoder.parameters(), lr=2e-4)
optimizerDecoder_Adam = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerEncoder_SGD = optim.SGD(encoder.parameters(), lr=2e-4)
optimizerEncoder_Adam = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)



#optimizerG = optim.RMSprop(netG.parameters(), lr=5e-5)

# container generate
fake_label = 0

if options.cuda:
    netD.cuda()
    decoder.cuda()
    encoder.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    criterion_MSE.cuda()

# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (pre_frame, real_mid_frame, nxt_frame) in enumerate(dataloader, 0):
        batch_size = real_mid_frame.size(0)
        optimizerEncoder_Adam.zero_grad()
        optimizerDecoder_Adam.zero_grad()

        original_real_mid_frame = Variable(real_mid_frame.clone().cuda())
        noised_real_mid_frame = torch.mul(real_mid_frame.cuda(), noise_mask(real_mid_frame.shape))
        pre_frame = Variable(pre_frame).cuda()
        nxt_frame = Variable(nxt_frame).cuda()
        
        pre_z = encoder(pre_frame)
        nxt_z = encoder(nxt_frame)
        mid_z = encoder(original_real_mid_frame)
        recon_pre_frame = decoder(pre_z)
        recon_nxt_frame = decoder(nxt_z)
        recon_mid_frame = decoder(mid_z)

        mse_err = criterion_MSE(recon_mid_frame, original_real_mid_frame.detach()) + \
                  criterion_MSE(recon_pre_frame, pre_frame.detach()) + \
                  criterion_MSE(recon_nxt_frame, nxt_frame.detach())
        mse_err.backward(retain_graph=True)

        optimizerEncoder_Adam.step()
        optimizerDecoder_Adam.step()

        '''
        #====================================
        optimizerEncoder_Adam.zero_grad()
        optimizerDecoder_Adam.zero_grad()
        optimizerD_Adam.zero_grad()


        z_fake = torch.cat((pre_z.view(pre_z.size(0), -1), nxt_z.view(nxt_z.size(0), -1), mid_z.view(mid_z.size(0), -1)), 1)
        d_fake = netD(z_fake)


        noise_pre = Variable(torch.FloatTensor(batch_size, nz)).cuda()
        noise_nxt = Variable(torch.FloatTensor(batch_size, nz)).cuda()
        noise_pre.data.normal_(0, 1)
        noise_nxt.data.normal_(0, 1)
        noise_mid = (noise_pre.detach()+noise_nxt.detach())/2

        z_real = torch.cat((noise_pre, noise_nxt, noise_mid), 1)
        d_real = netD(z_real)

        err_discriminator = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
        err_discriminator.backward(retain_graph=True)

        z_fake_2 = torch.cat(
            (pre_z.view(pre_z.size(0), -1), nxt_z.view(nxt_z.size(0), -1), mid_z.view(mid_z.size(0), -1)), 1)
        d_fake_2 = netD(z_fake_2)

        err_generator = -torch.mean(torch.log(d_fake_2))
        err_generator.backward(retain_graph=True)

        optimizerEncoder_Adam.step()
        optimizerDecoder_Adam.step()
        optimizerD_Adam.step()
        #====================================
        '''
        optimizerEncoder_Adam.zero_grad()
        optimizerDecoder_Adam.zero_grad()

        pre_z = encoder(pre_frame)
        nxt_z = encoder(nxt_frame)
        fake_mid_z = (pre_z.detach() + nxt_z.detach())/2
        recon_mid_frame = decoder(fake_mid_z)
        mse_err = criterion_MSE(recon_mid_frame, original_real_mid_frame.detach())
        mse_err.backward(retain_graph=True)

        optimizerEncoder_Adam.step()
        optimizerDecoder_Adam.step()
        

        #visualize
        print('[%d/%d][%d/%d] Loss_recon: %.4f'
             % (epoch, options.iteration, i, len(dataloader),
                mse_err.data.mean()))

        #if i == len(dataloader)-1:
        if True:
            testImage = torch.cat((unorm(pre_frame.data[0]),unorm(original_real_mid_frame.data[0]),unorm(recon_mid_frame.data[0]),unorm(nxt_frame.data[0])),2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Sandwich GAN"])
            line_win_dict =LJY_visualize_tools.draw_lines_to_windict(line_win_dict,[mse_err.data.mean(),0],
                                                                     ['loss_recon','zero'], epoch, i, len(dataloader))

            '''
            vutils.save_image(testImage,
                    '%s/%d_test_samples.png' % (options.outf,i),
                    normalize=True)
            '''

    # do checkpointing
    if IsSave == True:
        torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (options.outf, epoch))
        torch.save(decoder.state_dict(), '%s/decoder_epoch_%d.pth' % (options.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (options.outf, epoch))



# Je Yeol. Lee \[T]/
# Jolly Co-operation