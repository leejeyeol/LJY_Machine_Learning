import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
# import custom package
import LJY_utils
import LJY_visualize_tools

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png', help='path to dataset')
parser.add_argument('--fold', type=int, default=None, help = 'fold number')
parser.add_argument('--fold_dataroot', default='',help='Proprocessing/fold_divider.py')

parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=True, help='display options. default:False.')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=100, help='dimension of noise.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)


# Generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#150x150
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nz*1*1 => 512*4*4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 512*4*4 => 512*8*8
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 512*8*8 => 256*18*18
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 256*18*18 => 128*36*36
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 128*36*36 => 64*74*74
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64*74*74 => 3*150*150
            nn.ConvTranspose2d(ngf, nc, 4, 2, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

'''
#64x64
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
'''

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

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

class fold_Dataloader(torch.utils.data.Dataset):
    def __init__(self,fold, path, transform, type ='train'):
        super().__init__()
        self.transform = transform

        self.type = type
        train_path, val_path = LJY_utils.fold_loader(fold, path)
        if self.type == 'train':
            self.file_paths = train_path[0]
        # self.Semantic_base_path = Semantic_path
        elif self.type == 'validation':
            self.file_paths = val_path[0]

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
    #transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if options.fold is None:
    dataloader = torch.utils.data.DataLoader(fold_Dataloader.Dataloader(options.dataroot, transform),
                                             batch_size=options.batchSize, shuffle=True, num_workers=options.workers,drop_last=False)
else:
    dataloader = torch.utils.data.DataLoader(fold_Dataloader.fold_Dataloader(options.fold, options.fold_dataroot, transform, type='train'),
                                             batch_size=options.batchSize, shuffle=True, num_workers=options.workers,drop_last=False)
unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

# ======================================================================================================================
# Models
# ======================================================================================================================

# Generator ============================================================================================================
netG = Generator(ngpu, nz, ngf, nc)
netG.apply(LJY_utils.weights_init)
if options.netG != '':
    netG.load_state_dict(torch.load(options.netG))
print(netG)

# Discriminator ========================================================================================================
netD = Discriminator(ngpu, ndf, nc)
netD.apply(LJY_utils.weights_init)
if options.netD != '':
    netD.load_state_dict(torch.load(options.netD))
print(netD)

# ======================================================================================================================
# Training
# ======================================================================================================================

# criterion set
criterion_D = nn.BCELoss()
criterion_G = nn.BCELoss()


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerG = optim.Adam(netG.parameters(), betas=(0.5, 0.999), lr=1e-3)



# container generate
noise = torch.FloatTensor(batch_size, nz, 1, 1)

label = torch.FloatTensor(batch_size)
real_label = 1
fake_label = 0

if options.cuda:
    netD.cuda()
    netG.cuda()
    criterion_D.cuda()
    criterion_G.cuda()
    label = label.cuda()
    noise = noise.cuda()


# make to variables ====================================================================================================

label = Variable(label)
noise = Variable(noise)

# for visualize
win_dict = LJY_visualize_tools.win_dict()
line_win_dict = LJY_visualize_tools.win_dict()

# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (data) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        optimizerD.zero_grad()

        input = Variable(data, requires_grad=True)
        if options.cuda:
           input = input.cuda()
        label.data.resize_(input.size(0)).fill_(real_label)

        outputD = netD(input)
        errD_real = criterion_D(outputD, label)
        errD_real.backward()
        visual_D_real = outputD.data.mean()   # for visualize

        # generate noise    ============================================================================================
        noise.data.resize_(input.size(0), nz, 1, 1)
        noise.data.normal_(0, 1)

        # train with fake data   =======================================================================================
        fake = netG(noise)
        label.data.fill_(fake_label)

        outputD = netD(fake.detach())
        errD_fake = criterion_D(outputD, label)
        errD_fake.backward()
        visual_D_fake = outputD.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network and Q network
        ###########################
        optimizerG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost

        outputD = netD(fake)
        errG = criterion_G(outputD, label)
        errG.backward()
        visual_D_fake_2 = outputD.data.mean()

        optimizerG.step()

        #visualize
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f     D(x): %.4f D(G(z)): %.4f | %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 errD.data[0], errG.data[0],  visual_D_real, visual_D_fake, visual_D_fake_2))

        if display:
            testImage = torch.cat((unorm(input.data[0]), unorm(fake.data[0])), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["DCGAN_%s" % dataset])
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [errD.data.mean(), errG.data.mean(), visual_D_real,
                                                                       visual_D_fake],
                                                                      ['lossD', 'lossG', 'real is?', 'fake is?'], epoch,i,
                                                                      len(dataloader))

    # do checkpointing
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), '%s/%d_fold_netG_epoch_%d.pth' % (options.outf, options.fold, epoch))
        torch.save(netD.state_dict(), '%s/%d_fold_netD_epoch_%d.pth' % (options.outf, options.fold, epoch))

# Je Yeol. Lee \[T]/