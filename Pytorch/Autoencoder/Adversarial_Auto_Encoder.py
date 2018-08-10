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
import torchvision.utils as vutils
from torch.autograd import Variable
import LJY_utils
import LJY_visualize_tools

import matplotlib.pyplot as plt
import math

criterion = nn.MSELoss(size_average=False).cuda()
def Variational_loss(input, target, mu, logvar):
    recon_loss = criterion(input, target)
    KLD_loss = -0.5 * torch.sum(1+logvar-mu.pow(2) - logvar.exp())
    print(KLD_loss)
    return recon_loss + KLD_loss

class encoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=2, num_filters=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 4 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * num_filters, z_size, 3, 1, 0),
        )
        self.fc_mu = nn.Conv2d(z_size, z_size, 1)
        self.fc_sig = nn.Conv2d(z_size, z_size, 1)
        # init weights
        self.weight_init()

    def forward(self,x):
        #VAE
        z_ = self.encoder(x)
        mu = self.fc_mu(z_)
        logvar = self.fc_sig(z_)
        return mu, logvar

    '''
    def forward(self, x):
        #AE
        z = self.encoder(x)
        return z
    '''
    def weight_init(self):
        self.encoder.apply(weight_init)

class decoder(nn.Module):
    def __init__(self, num_in_channels=1, z_size=2, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 5, 1, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, 5, 1, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, 5, 2, 0),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 0),
            nn.Tanh()
        )
        # init weights
        self.weight_init()

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def weight_init(self):
        self.decoder.apply(weight_init)

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(0.2, True),

            nn.Linear(100, 50),
            nn.LeakyReLU(0.2, True),

            nn.Linear(50, 30),
            nn.LeakyReLU(0.2, True),

            nn.Linear(30, 1),
            nn.Sigmoid()
        )

        # init weights
        self.weight_init()
    def forward(self, z):
        cls = self.discriminator(z)

        return cls

    def weight_init(self):
        self.discriminator.apply(weight_init)
# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

class unbiased_MNIST(dset.MNIST):
    def __init__(self, root,  train, download, transform):
        super().__init__(root=root,  train=train, download=download, transform=transform)
        idx_per_label = [[] for _ in range(10)]
        label_list = list(self.train_labels)
        for i in range(len(label_list)):
            idx_per_label[label_list[i]].append(i)
        random.seed(1)
        # 0 : 5923 => 5923
        # 1 : 6742 => 2000
        # 2 : 5958 => 2000
        # 3 : 6131 => 2000
        # 4 : 5842 => 2000
        # 5 : 5421 => 500
        # 6 : 5918 => 500
        # 7 : 6265 => 500
        # 8 : 5851 => 500
        # 9 : 5949 => 500

        # container
        unbiased_train_data = self.train_data[0].view(1, 28, 28)
        unbiased_labels = [self.train_labels[0]]

        # sampling data
        for idx in idx_per_label[0]:
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[1], 2000):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[2], 2000):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[3], 2000):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[4], 2000):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[5], 500):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[6], 500):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[7], 500):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[8], 500):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])
        for idx in random.sample(idx_per_label[9], 500):
            torch.cat((unbiased_train_data, self.train_data[idx].view(1, 28, 28)), 0, unbiased_train_data)
            unbiased_labels.append(self.train_labels[idx])

        self.train_labes = torch.LongTensor(unbiased_labels)
        self.train_data = unbiased_train_data





#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default='/mnt/fastdataset/Datasets', help='path to dataset')
parser.add_argument('--netG', default='', help="path of Generator networks.(to continue training)")
parser.add_argument('--netD', default='', help="path of Discriminator networks.(to continue training)")
parser.add_argument('--outf', default='./pretrained_model', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=12000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=800, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=2, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

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




ngpu = int(options.ngpu)
nz = int(options.nz)

encoder = encoder()
encoder.apply(LJY_utils.weights_init)
if options.netG != '':
    encoder.load_state_dict(torch.load(options.netG))
print(encoder)

decoder = decoder()
decoder.apply(LJY_utils.weights_init)
if options.netD != '':
    decoder.load_state_dict(torch.load(options.netD))
print(decoder)

discriminator = discriminator()
discriminator.apply(LJY_utils.weights_init)
if options.netD != '':
    discriminator.load_state_dict(torch.load(options.netD))
print(discriminator)
#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()

# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)


# container generate
input = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize)


if options.cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()
    input = input.cuda()



# make to variables ====================================================================================================
input = Variable(input)


# training start
def train():
    ep = 0
    '''
    encoder.load_state_dict(torch.load(os.path.join(options.outf, "weighted_unbiased_encoder_epoch_%d.pth") % ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "weighted_unbiased_decoder_epoch_%d.pth") % ep))
    discriminator.load_state_dict(
        torch.load(os.path.join(options.outf, "weighted_unbiased_discriminator_epoch_%d.pth") % ep))
    '''
    dataloader = torch.utils.data.DataLoader(
        unbiased_MNIST(root='../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")
    for epoch in range(options.iteration):
        for i, (data, _) in enumerate(dataloader, 0):
            real = Variable(torch.Tensor(data.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
            fake = Variable(torch.Tensor(data.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

            ############################
            # (1) Update D network
            ###########################
            # autoencoder training  ========================================================================================
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)


            #z = encoder(input)

            mu, logvar = encoder(input)
            std = torch.exp(0.5 * logvar)
            eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
            z = eps.mul(std).add_(mu)

            x_recon = decoder(z)

            err_mse = MSE_loss(x_recon, input.detach())
            err_mse.backward(retain_graph=True)
            err_image = (input.detach()-x_recon.detach()).abs()

            optimizerE.step()
            optimizerD.step()



            # discriminator training =======================================================================================
            optimizerDiscriminator.zero_grad()

            d_fake = discriminator(z.view(batch_size, 2))

            noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1).cuda())
            noise.data.normal_(0, 1)
            d_real = discriminator(noise.view(batch_size, 2))


            err_discriminator = (BCE_loss(d_real,real) + BCE_loss(d_fake,fake))
            err_discriminator.backward(retain_graph=True)
            optimizerDiscriminator.step()

            # generator = encoder training =================================================================================
            optimizerE.zero_grad()

            d_fake_2 = discriminator(z.view(batch_size, 2))
            err_generator = BCE_loss(d_fake_2, real)
            err_generator.backward(retain_graph=True)

            gaussian_entropy = 1/2*math.log(2*math.pi*math.exp(1))
            err_anomaly = torch.abs(torch.mean(z**2/2+1/2*math.log(2*math.pi))-gaussian_entropy)
            err_anomaly.backward(retain_graph=True)
            optimizerE.step()

            #visualize
            print('[%d/%d][%d/%d] Loss: %.4f anomaly : %.4f'% (epoch, options.iteration, i, len(dataloader), err_mse.data.mean(), err_anomaly.data.mean()))
            testImage = torch.cat((input.data[0], x_recon.data[0]), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage, err_image.data[0]], ["Autoencoder","error_image"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_mse.data.mean(), d_real.data.mean(),
                                                                       d_fake.data.mean(), err_anomaly.data.mean(),
                                                                       err_discriminator.data.mean(), err_generator.data.mean(),
                                                                       0],
                                                                      ['loss_recon_x', 'err_real d',
                                                                       'fake d','anomaly','D loss','G loss', 'zero'],
                                                                      epoch, i, len(dataloader))



        # do checkpointing
        if epoch % 50 == 0:
            torch.save(encoder.state_dict(), '%s/weighted_unbiased_encoder_epoch_%d.pth' % (options.outf, epoch+ep))
            torch.save(decoder.state_dict(), '%s/weighted_unbiased_decoder_epoch_%d.pth' % (options.outf, epoch+ep))
            torch.save(discriminator.state_dict(), '%s/weighted_unbiased_discriminator_epoch_%d.pth' % (options.outf, epoch+ep))

def test():
    ep = 200
    encoder.load_state_dict(torch.load(os.path.join(options.outf, "weighted_unbiased_encoder_epoch_%d.pth")%ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "weighted_unbiased_decoder_epoch_%d.pth")%ep))
    discriminator.load_state_dict(torch.load(os.path.join(options.outf, "weighted_unbiased_discriminator_epoch_%d.pth")%ep))

    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=1, shuffle=False, num_workers=options.workers)
    vis_x = []
    vis_y = []
    vis_label = []
    print("Testing Start!")
    for i, (data, label) in enumerate(dataloader, 0):
        real_cpu = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        #z = encoder(input)
        mu, logvar = encoder(input)
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
        z = eps.mul(std).add_(mu)

        vis_x.append(z.data.view(2)[0])
        vis_y.append(z.data.view(2)[1])
        vis_label.append(int(label))
        print("[%d/%d]"%(i,len(dataloader)))

    fig = plt.figure()
    plt.scatter(vis_x,vis_y,c=vis_label, s=2, cmap='tab10')
    cb = plt.colorbar()
    plt.show()

if __name__ == "__main__" :
    train()
    #test()



# Je Yeol. Lee \[T]/
# Jolly Co-operation