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
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math


class generator(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 5, 1, 1),
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
    def __init__(self, num_in_channels=1, z_size=1, num_filters=64):
        super().__init__()
        self.discriminator = nn.Sequential(
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
            nn.Sigmoid()
        )
        # init weights
        self.weight_init()

    def forward(self, x):
        d = self.discriminator(x)
        return d

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


# =======================================================================================================================
# Options
# =======================================================================================================================
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
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
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

# =======================================================================================================================
# Data and Parameters
# =======================================================================================================================

# MNIST call and load   ================================================================================================


ngpu = int(options.ngpu)
nz = int(options.nz)

generator = generator(num_in_channels=1, z_size=nz, num_filters=64)
generator.apply(LJY_utils.weights_init)
if options.netD != '':
    generator.load_state_dict(torch.load(options.netD))
print(generator)

discriminator = discriminator(num_in_channels=1, z_size=1, num_filters=64)
discriminator.apply(LJY_utils.weights_init)
if options.netD != '':
    discriminator.load_state_dict(torch.load(options.netD))
print(discriminator)
# =======================================================================================================================
# Training
# =======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()

# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerGenerator = optim.Adam(generator.parameters(), betas=(0.5, 0.999), lr=2e-3)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=2e-4)

if options.cuda:
    generator.cuda()
    discriminator.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()


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
        dset.MNIST(root='../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
    unorm = LJY_visualize_tools.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")
    for epoch in range(options.iteration):
        for i, (data, _) in enumerate(dataloader, 0):

            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()
            disc_input = input.clone()

            real_label = Variable(torch.FloatTensor(batch_size, 1, 1, 1).cuda())
            real_label.data.fill_(1)
            fake_label = Variable(torch.FloatTensor(batch_size, 1, 1, 1).cuda())
            fake_label.data.fill_(0)

            # z = encoder(input)

            alpha = 0.9


            '''
            # visualize
            print('[%d/%d][%d/%d] MSE_Loss: %.4f KLD_Loss: %.4f'
                  % (epoch, options.iteration, i, len(dataloader), err_recon.data.mean(),err_KLD.data.mean()))
            testImage = torch.cat((unorm(input.data[0]), unorm(x_recon.data[0])), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_recon.data.mean(),
                                                                       err_KLD.data.mean(),
                                                                       0],
                                                                      ['loss_recon_x',
                                                                        'loss_KLD',
                                                                       'zero'],
                                                                      epoch, i, len(dataloader))

            '''
            # discriminator training =======================================================================================
            optimizerDiscriminator.zero_grad()
            optimizerGenerator.zero_grad()
            d_real = discriminator(disc_input)

            noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = generator(noise)
            d_fake = discriminator(generated_fake)
            balance_coef = torch.cat((d_fake, d_real), 0).mean()
            err_discriminator_origin = (1 - alpha) * (BCE_loss(d_real, real_label)
                                                      + BCE_loss(d_fake, fake_label))
            err_discriminator = err_discriminator_origin + BCE_loss(balance_coef, Variable(torch.FloatTensor(balance_coef.shape).fill_(0.5).cuda()))

            err_discriminator.backward(retain_graph=True)
            optimizerDiscriminator.step()

            noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = generator(noise)
            d_fake_2 = discriminator(generated_fake)

            err_generator = (1 - alpha) * (BCE_loss(d_fake_2, real_label))
            err_generator.backward(retain_graph=True)
            optimizerGenerator.step()

            # visualize
            print('[%d/%d][%d/%d] d_real: %.4f d_fake: %.4f Balance : %.2f'
                  % (epoch, options.iteration, i, len(dataloader), d_real.data.mean(), d_fake_2.data.mean(), balance_coef.data.mean()))
            testImage = torch.cat((unorm(input.data[0]), unorm(generated_fake.data[0])), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [err_discriminator_origin.data.mean(),
                                                                       err_generator.data.mean(),
                                                                       balance_coef.data[0],
                                                                       0],
                                                                      ['D loss',
                                                                       'G loss',
                                                                       'balance',
                                                                       'zero'],
                                                                      epoch, i, len(dataloader))

        # do checkpointing
        if epoch % 1000 == 0:
            torch.save(generator.state_dict(), '%s/fcn_anomaly_generator_epoch_%d.pth' % (options.outf, epoch + ep))
            torch.save(discriminator.state_dict(),
                       '%s/fcn_anomaly_discriminator_epoch_%d.pth' % (options.outf, epoch + ep))


def tsne():
    ep = 6000
    encoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_encoder_epoch_%d.pth") % ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_decoder_epoch_%d.pth") % ep))
    discriminator.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_discriminator_epoch_%d.pth") % ep))

    dataloader = torch.utils.data.DataLoader(
        dset.MNIST('../../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=1, shuffle=False, num_workers=options.workers)

    # vis_x = []
    # vis_y = []
    vis_label = []
    vis = []
    print("Testing Start!")
    for i, (data, label) in enumerate(dataloader, 0):
        real_cpu = data
        batch_size = real_cpu.size(0)
        input = Variable(real_cpu).cuda()

        z = encoder(input)
        '''
        mu, logvar = encoder(input)
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
        z = eps.mul(std).add_(mu)
        '''

        vis.append(np.asarray(z.data.view(nz)))
        # vis_x.append(z.data.view(nz))
        # vis_y.append(z.data.view(nz))

        vis_label.append(int(label))
        print("[%d/%d]" % (i, len(dataloader)))
    vis = np.asarray(vis)
    z_embedded = TSNE(n_components=2).fit_transform(vis)


def test():
    ep = 500

    encoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_encoder_epoch_%d.pth") % ep))
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_decoder_epoch_%d.pth") % ep))
    discriminator.load_state_dict(
        torch.load(os.path.join(options.outf, "fcn_anomaly_discriminator_epoch_%d.pth") % ep))

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
        input = Variable(real_cpu).cuda()

        z = encoder(input)
        '''
        mu, logvar = encoder(input)
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
        z = eps.mul(std).add_(mu)
        '''

        vis_x.append(z.data.view(nz)[0])
        vis_y.append(z.data.view(nz)[1])

        vis_label.append(int(label))
        print("[%d/%d]" % (i, len(dataloader)))
    fig = plt.figure()
    plt.scatter(vis_x, vis_y, c=vis_label, s=2, cmap='tab10')
    cb = plt.colorbar()
    plt.show()


def visualize_latent_space():
    ep = 6000
    decoder.load_state_dict(torch.load(os.path.join(options.outf, "fcn_anomaly_decoder_epoch_%d.pth") % ep))
    x = range(-40, 40)
    y = range(-40, 40)
    image = torch.FloatTensor()
    for i in range(len(x)):
        x_image = torch.FloatTensor()
        for j in range(len(y)):
            z = Variable(torch.FloatTensor(1, 2)).cuda()
            z.data[0][0] = x[i]
            z.data[0][1] = y[j]
            recon_x = decoder(z)
            x_image = torch.cat((x_image, recon_x.view(1, 28, 28)), 2)
            image = torch.cat((image, x_image), 3)
    print(1)


if __name__ == "__main__":
    train()
    # test()
    # tsne()
    # visualize_latent_space()

# Je Yeol. Lee \[T]/
# Jolly Co-operation.tolist()