import torch.utils.data as ud
import argparse

import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import LJY_utils
import LJY_visualize_tools

from PIL import Image
from PIL import ImageOps
import os
import glob
import torch.utils.data

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='celebA', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/img', help='path to dataset')
parser.add_argument('--pretrainedModelName', default='autoencoder', help="path of Encoder networks.")
parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks.")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=10000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=200, help='input batch size')
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

def Variational_loss(input, target, mu, logvar):
    recon_loss = MSE_loss(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    return recon_loss + KLD_loss

class CelebA_DL(torch.utils.data.Dataset):
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
                return img.convert('L')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        path = self.file_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class DL(torch.utils.data.Dataset):
    def __init__(self, path, transform,type):
        random.seed = 1
        super().__init__()
        self.transform = transform
        self.type = type
        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()
        total_file_paths = []
        # training set ================
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE_normal', '*'))
        total_file_paths = total_file_paths+cur_file_paths

        cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        #cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
        #total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        random.shuffle(total_file_paths)

        #num_of_valset = 1
        #self.val_file_paths = sorted(total_file_paths[:num_of_valset])
        #self.file_paths = sorted(total_file_paths[num_of_valset:])
        self.file_paths = sorted(total_file_paths)

        total_file_paths = []
        #test set ===========================
        '''
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'CMU_PIE', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'YaleB', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'Yale', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'AR', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        '''
        #for occlusion
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET', '*'))
        total_file_paths = total_file_paths + cur_file_paths
        cur_file_paths = glob.glob(os.path.join(self.base_path, 'FERET_normal', '*'))
        total_file_paths = total_file_paths + cur_file_paths

        # for testset
        self.val_file_paths = sorted(total_file_paths)

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = ImageOps.equalize(img)
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
    def __init__(self,  z_size=2, channel=3, num_filters=64, type='AE'):
        super().__init__()
        self.type = type
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
        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(z_size, z_size, 1)
            self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self, x):
        if self.type == 'AE':
            #AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar


    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder(nn.Module):
    def __init__(self, z_size=2, channel=3, num_filters=64):
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
        '''
        self.decoder_subpixel = nn.Sequential(
        )
        '''
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
autoencoder_type = 'AE'
encoder = encoder(options.nz, options.nc, 64, autoencoder_type)
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
MSE_loss = nn.MSELoss(size_average=False)

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
        batch_size=options.batchSize, shuffle=True, drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(
        DL(options.dataroot, transform, 'test'),
        batch_size=options.batchSize, shuffle=True)
    '''
    celebA dataset
    transform_celebA = transforms.Compose([
        transforms.CenterCrop(150),
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    celebA_dataroot = '/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png'
    dataloader_celebA = torch.utils.data.DataLoader(CelebA_DL(celebA_dataroot, transform_celebA),
                                             batch_size=options.batchSize, shuffle=True, drop_last=False)
    '''
    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    line_win_dict_train_val = LJY_visualize_tools.win_dict()
    print(autoencoder_type)
    print("Training Start!")
    ep = 0
    if ep != 0 :
        encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_VAE_encoder_epoch_%d.pth") % ep))
        decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_VAE_decoder_epoch_%d.pth") % ep))

    for epoch in range(options.iteration):
        train_err = 0
        for i, data in enumerate(dataloader, 0):
            # autoencoder training  ====================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            original_data = Variable(real_cpu).cuda()
            input.data.resize_(real_cpu.size()).copy_(make_random_squre(real_cpu))
            #input.data.resize_(real_cpu.size()).copy_(real_cpu)

            if autoencoder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err_mse = MSE_loss(x_recon, original_data.detach())
            elif autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_mse = Variational_loss(x_recon, original_data.detach(), mu, logvar)



            err_mse.backward(retain_graph=True)
            train_err += float(err_mse.data.mean())

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
        train_err = (train_err/i)
        val_err = 0
        for i, data in enumerate(dataloader_val, 0):
            # autoencoder training  ====================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)

            original_data = Variable(real_cpu).cuda()
            input.data.resize_(real_cpu.size()).copy_(make_random_squre(real_cpu))
            # input.data.resize_(real_cpu.size()).copy_(real_cpu)

            if autoencoder_type == 'AE':
                z = encoder(input)
                x_recon = decoder(z)
                err_mse = MSE_loss(x_recon, original_data.detach())
            elif autoencoder_type == 'VAE':
                mu, logvar = encoder(input)
                std = torch.exp(0.5 * logvar)
                eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                z = eps.mul(std).add_(mu)
                x_recon = decoder(z)
                err_mse = Variational_loss(x_recon, original_data.detach(), mu, logvar)

            val_err += float(err_mse.data.mean())

        val_err = (val_err/i)
        line_win_dict_train_val = LJY_visualize_tools.draw_lines_to_windict(line_win_dict_train_val,
                                                                  [train_err, val_err, 0],
                                                                  ['train_err_per_epoch', 'val_err_per_epoch','zero'],
                                                                            0, epoch, options.iteration)
        # do checkpointing
        if epoch % 100 == 0:
            torch.save(encoder.state_dict(), '%s/face_noFERET_VAE_encoder_epoch_%d.pth' % (options.outf, ep+epoch))
            torch.save(decoder.state_dict(), '%s/face_noFERET_VAE_decoder_epoch_%d.pth' % (options.outf, ep+epoch))

def test():
    in_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    LJY_utils.make_dir("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result")

    print("Testing Start!")
    trained = 1261
    for j in range(trained):
        ep = (j)*9 + 1
        encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_encoder_epoch_%d.pth")%ep))
        decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_decoder_epoch_%d.pth")%ep))

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

            print("[%d/%d]" % (ep, trained))

def test_img():
    win_dict = LJY_visualize_tools.win_dict()

    print("Testing Start!")

    LJY_utils.make_dir("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result")

    ep = 7000


    encoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_encoder_epoch_%d.pth") % ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "face_noFERET_decoder_epoch_%d.pth") % ep))

    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    dataloader = torch.utils.data.DataLoader(DL(options.dataroot, transform,'test'),
                                             batch_size=1, shuffle=False)
    mse_box = 0
    mse_nobox = 0
    for i, data in enumerate(dataloader, 0):
        real_cpu = data

        original_data = Variable(real_cpu).cuda()
        if autoencoder_type == 'AE':
            z = encoder(original_data)
        elif autoencoder_type == 'VAE':
            mu, logvar = encoder(original_data)
            std = torch.exp(0.5 * logvar)
            eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
            z = eps.mul(std).add_(mu)
        x_recon = decoder(z)


        err_mse_nobox = MSE_loss(x_recon, original_data.detach())
        mse_nobox += float(err_mse_nobox.data)

        input_bbox = Variable(make_random_squre(real_cpu)).cuda()
        if autoencoder_type == 'AE':
            z = encoder(input_bbox)
        elif autoencoder_type == 'VAE':
            mu, logvar = encoder(input_bbox)
            std = torch.exp(0.5 * logvar)
            eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
            z = eps.mul(std).add_(mu)
        x_recon_bbox = decoder(z)
        err_mse_box = MSE_loss(x_recon_bbox, input_bbox.detach())
        mse_box += float(err_mse_box.data)
        dif_img = x_recon.detach() - x_recon_bbox.detach()


        testImage = torch.cat((unorm(original_data.data[0]), unorm(x_recon.data[0]), unorm(input_bbox.data[0]), unorm(x_recon_bbox.data[0]), unorm(dif_img.data[0])), 2)
        toimg = transforms.ToPILImage()
        toimg(testImage.cpu()).save("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result"+"/%05d.png"%i)
        #win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

        print("[%d/%d]"%(i,len(dataloader)))
    os.rename("/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/result","/media/leejeyeol/74B8D3C8B8D38750/Data/face_autoencoder/"+"result_FARET_%d_%.2f_%.2f"%(ep, mse_nobox / len(dataloader),mse_box/len(dataloader)))


if __name__ == "__main__" :
    #train()
    #test()
    test_img()



# Je Yeol. Lee \[T]/
# Jolly Co-operation