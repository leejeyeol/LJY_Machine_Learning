import torch.utils.data as ud
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

import math
import glob as glob

from PIL import Image
import matplotlib.pyplot as plt

import LJY_utils
import LJY_visualize_tools

plt.style.use('ggplot')

#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='HMDB51', help='what is dataset? MG : Mixtures of Gaussian', choices=['CelebA', 'MNIST', 'MG'])
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/flow/flow', help='path to dataset')

parser.add_argument('--pretrainedEpoch', type=int, default=0, help="path of Decoder networks. '0' is training from scratch.")
parser.add_argument('--pretrainedModelName', default='HMDB_OF', help="path of Encoder networks.")
parser.add_argument('--modelOutFolder', default='./pretrained_model', help="folder to model checkpoints")
parser.add_argument('--resultOutFolder', default='./results', help="folder to test results")
parser.add_argument('--save_tick', type=int, default=1, help='save tick')
parser.add_argument('--display_type', default='per_epoch', help='displat tick',choices=['per_epoch', 'per_iter'])

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--save', default=False, help='save options. default:False. NOT IMPLEMENTED')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=15000, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=512, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

# custom options
parser.add_argument('--netQ', default='', help="path of Auxiliaty distribution networks.(to continue training)")

options = parser.parse_args()
print(options)


class custom_Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()
        cur_file_paths = []
        HMDB_action_folders = sorted(glob.glob(self.base_path + '/*'))
        for HMDB_actions in HMDB_action_folders:
            HMDB_action = sorted(glob.glob(HMDB_actions + '/*'))
            for clips in HMDB_action:
                clip = sorted(glob.glob(clips + '/*'))
                cur_file_paths = cur_file_paths + clip

        print("data loading complete!")
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

        return img, 1


def pepper_noise(ins, is_training, prob = 0.9):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        return torch.mul(ins, mask)
    return ins

def gaussian_noise(ins, is_training, mean=0, stddev=1,prob = 0.9):
    if is_training:
        mask = torch.Tensor(ins.shape).fill_(prob)
        mask = torch.bernoulli(mask)
        mask = Variable(mask)
        if ins.is_cuda is True:
            mask = mask.cuda()
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))/255
        noise = torch.mul(noise, mask)
        return ins + noise
    return ins

class generator(nn.Module):
    '''Generator'''
    def __init__(self):
        super(generator, self).__init__()

        self.deconv4 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0,  bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.deconv6 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=0,  bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.deconv4(h)
        h = self.bn4(h)
        h = self.relu4(h)  # 512,3,3

        h = self.deconv5(h)
        h = self.bn5(h)
        h = self.relu5(h)  # 512,7,7

        h = self.deconv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,14,14

        h = self.deconv7(h)
        h = self.bn7(h)
        h = self.relu7(h) # 256,28,28

        h = self.deconv8(h)
        h = self.bn8(h)
        h = self.relu8(h) # 128,56,56

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h) # 64,112,112

        h = self.deconv10(h)
        h = F.tanh(h) # 3,224,224

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class discriminator(nn.Module):
    '''Discriminator'''
    def __init__(self, large=False):
        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        if large:
            self.conv6 = nn.Conv2d(512, 512, 15, stride=1, padding=0, bias=False)
        else:
            self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h) # 64,112,112 (if input is 224x224)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h) # 128,56,56

        h = self.conv3(h) # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h) # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h) # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h) # 512,1,1

        h = self.conv7(h)
        h = F.sigmoid(h)

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)





# save directory make   ================================================================================================
try:
    os.makedirs(options.modelOutFolder)
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

generator = generator()
print(generator)

discriminator = discriminator()
print(discriminator)




#=======================================================================================================================
# Training
#=======================================================================================================================

# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss(size_average=False)

# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerGenerator = optim.Adam(generator.parameters(), betas=(0.5, 0.999), lr=2e-4)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=2e-3)

if options.cuda:
    generator.cuda()
    discriminator.cuda()
    MSE_loss.cuda()
    BCE_loss.cuda()


# training start
def train():
    ep = options.pretrainedEpoch
    if ep != 0:
        generator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_generator" + "_%d.pth" % ep)))
        discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))

    dataloader = torch.utils.data.DataLoader(
        custom_Dataloader(path=options.dataroot,
                          transform=transforms.Compose([
                              transforms.Scale((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
                   ])),batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    unorm = LJY_visualize_tools.UnNormalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225))

    win_dict = LJY_visualize_tools.win_dict()
    line_win_dict = LJY_visualize_tools.win_dict()
    grad_line_win_dict = LJY_visualize_tools.win_dict()
    print("Training Start!")

    alpha = 0.5
    for epoch in range(options.epoch):
        for i, (data, _) in enumerate(dataloader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()
            disc_input =input.clone()

            real_label = Variable(torch.FloatTensor(batch_size, 1,1,1).cuda())
            real_label.data.fill_(1)
            fake_label = Variable(torch.FloatTensor(batch_size, 1,1,1).cuda())
            fake_label.data.fill_(0)

            # discriminator training =======================================================================================
            optimizerDiscriminator.zero_grad()
            d_real = discriminator(disc_input)
            noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = generator(noise)
            d_fake = discriminator(generated_fake)

            err_discriminator_real = BCE_loss(d_real, real_label)
            err_discriminator_fake = BCE_loss(d_fake, fake_label)

            err_discriminator_origin = err_discriminator_real + err_discriminator_fake
            err_discriminator = err_discriminator_origin
            err_discriminator.backward(retain_graph=True)
            discriminator_grad = LJY_utils.torch_model_gradient(discriminator.parameters())
            optimizerDiscriminator.step()

            optimizerGenerator.zero_grad()
            noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
            noise.data.normal_(0, 1)
            generated_fake = generator(noise)
            d_fake_2 = discriminator(generated_fake)
            err_generator = BCE_loss(d_fake_2, real_label)
            err_generator = (1-alpha) * err_generator
            err_generator.backward(retain_graph=True)

            generator_grad = LJY_utils.torch_model_gradient(generator.parameters())
            optimizerGenerator.step()
             #visualize
            print('[%d/%d][%d/%d] recon_Loss: GAN  d_real: %.4f d_fake: %.4f alpha : %.2f'
                  % (epoch, options.epoch, i, len(dataloader), d_real.data.mean(), d_fake_2.data.mean(),alpha))
            if options.display:
                testImage = torch.cat((unorm(input.data[0]), unorm(generated_fake.data[0])), 2)
            win_dict = LJY_visualize_tools.draw_images_to_windict(win_dict, [testImage], ["Autoencoder"])

            if options.display_type == 'per_iter':
                line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                          [
                                                                              err_discriminator_real.data.mean(),
                                                                              err_discriminator_fake.data.mean(),
                                                                              err_generator.data.mean(),
                                                                              0],
                                                                          [
                                                                              'D loss -real',
                                                                              'D loss -fake',
                                                                              'G loss',
                                                                              'zero'],
                                                                          epoch, i, len(dataloader))
                grad_line_win_dict = LJY_visualize_tools.draw_lines_to_windict(grad_line_win_dict,
                                                                               [
                                                                                   discriminator_grad,
                                                                                   generator_grad,
                                                                                   0],
                                                                               ['D gradient',
                                                                                'G gradient',
                                                                                'zero'],
                                                                               epoch, i, len(dataloader))

        if options.display_type =='per_epoch':
            line_win_dict = LJY_visualize_tools.draw_lines_to_windict(line_win_dict,
                                                                      [
                                                                       #z_err.data.mean(),
                                                                       err_discriminator_real.data.mean(),
                                                                       err_discriminator_fake.data.mean(),
                                                                       err_generator.data.mean(),
                                                                       0],
                                                                      [
                                                                       #'D_z',
                                                                       'D loss -real',
                                                                       'D loss -fake',
                                                                       'G loss',
                                                                       'zero'],
                                                                      0, epoch, 0)
            grad_line_win_dict = LJY_visualize_tools.draw_lines_to_windict(grad_line_win_dict,
                                                                      [
                                                                          discriminator_grad,
                                                                          generator_grad,
                                                                          0],
                                                                      [   'D gradient',
                                                                          'G gradient',
                                                                          'zero'],
                                                                      0, epoch, 0)

        # do checkpointing

        if epoch % options.save_tick == 0 or options.save:
            torch.save(generator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_generator" + "_%d.pth" % (epoch+ep)))
            torch.save(discriminator.state_dict(), os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % (epoch+ep)))
            print(os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch+ep)))

if __name__ == "__main__" :
    train()




# Je Yeol. Lee \[T]/
# Jolly Co-operation.tolist()