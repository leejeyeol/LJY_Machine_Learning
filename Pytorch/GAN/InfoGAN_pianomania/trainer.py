#base : https://github.com/pianomania/infoGAN-pytorch/blob/master/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np


class log_gaussian:

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:

    def __init__(self, G, FE, D, Q, E, nz, nc):

        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.E = E

        self.batch_size = 100
        self.nc = nc
        self.nz = nz

    def _noise_sample(self, con_c, noise):


        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, con_c], 1).view(-1, 64, 1, 1)

        return z

    def train(self):

        real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
        label = torch.FloatTensor(self.batch_size, 1).cuda()
        con_c = torch.FloatTensor(self.batch_size, self.nc).cuda()
        noise = torch.FloatTensor(self.batch_size, self.nz).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=0.0002,
                            betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=0.001,
                            betas=(0.5, 0.99))

        dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)


        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):

                # real part
                optimD.zero_grad()

                x, _ = batch_data

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                con_c.data.resize_(bs, self.nc)
                noise.data.resize_(bs, self.nz)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z = self._noise_sample(con_c, noise)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_mu, q_var = self.Q(fe_out)
                con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.1

                G_loss = reconstruct_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )
                    # fixed random variables
                    c = np.linspace(-1, 1, 10).reshape(1, -1)
                    c = np.repeat(c, 10, 0).reshape(-1, 1)

                    condition = []

                    for test_num in range(self.nc):
                        test_c_code = np.hstack([np.zeros_like(c) if i==test_num else c for i in range(self.nc)])
                        condition.append(test_c_code)
                    #c1 = np.hstack([c, np.zeros_like(c)])
                    #c2 = np.hstack([np.zeros_like(c), c])

                    fix_noise = []
                    for i in range(10):
                        tmp_noise = torch.Tensor(1, self.nz).uniform_(-1, 1)
                        tmp_fix_noise = torch.cat([tmp_noise for _ in range(10)], 0)
                        fix_noise.append(tmp_fix_noise)
                    fix_noise = torch.cat(fix_noise, 0)

                    noise.data.copy_(fix_noise)

                    for i in range(self.nc):
                        con_c.data.copy_(torch.from_numpy(condition[i]))
                        z = torch.cat([noise, con_c], 1).view(-1, self.nz + self.nc, 1, 1)
                        x_save = self.G(z)
                        save_image(x_save.data, './tmp/c%d.png'%i, nrow=10)