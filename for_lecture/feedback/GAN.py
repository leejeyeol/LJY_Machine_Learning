import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision.utils as vutils

def on_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    one_hot = y[labels]
    one_hot = y[labels]
    return one_hot


def softmax_to_one_hot(tensor):
    max_idx = torch.argmax(tensor, 1, keepdim=True)
    if tensor.is_cuda:
        one_hot = torch.zeros(tensor.shape).cuda()
    else:
        one_hot = torch.zeros(tensor.shape)
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def show_generated_data(real_data, fake_data):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_data[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_data.detach()[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


class MNIST_Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.nz = nz

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.nz, 256, 5, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        x_ = self.decoder(z)
        return x_


class MNIST_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 3, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_net(x)
        return y


epochs = 5
generator_learning_rate, discriminator_learning_rate = 0.001, 0.0002
batch_size = 100
nz = 2
fake_label = 0
real_label = 1
loss_function = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = datasets.MNIST('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ]))
num_train = len(dataset)
valid_size = 500

indices = list(range(num_train))
split = num_train - valid_size
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                       , transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

net_generator = MNIST_Generator(nz).cuda()
net_generator.apply(weight_init)

net_discriminator = MNIST_Discriminator().cuda()

generator_optimizer = optim.Adam(net_generator.parameters(), lr=generator_learning_rate)
discrimiantor_optimaizer = optim.Adam(net_discriminator.parameters(), lr=discriminator_learning_rate)

noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
# fake_data = net_generator(noise)
label = torch.FloatTensor(batch_size).to(device)
label.data.fill_(fake_label)

net_generator.train()
net_discriminator.train()
for epoch in range(epochs):
    for i, (X, _) in enumerate(train_loader):
        X = X.cuda()
        noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
        fake_data = net_generator(noise)
        label.data.fill_(fake_label)

        Y = net_discriminator(fake_data.detach())
        loss = loss_function(Y, label)
        discrimiantor_optimaizer.zero_grad()
        loss.backward()

        Y = net_discriminator(X)
        label = torch.FloatTensor(batch_size).to(device)
        label.data.fill_(real_label)
        loss = loss_function(Y, label)
        loss.backward()

        discrimiantor_optimaizer.step()

        noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
        fake_data = net_generator(noise)
        label.data.fill_(real_label)

        Y = net_discriminator(X)
        loss = loss_function(Y, label)
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
    print("[%d/%d][%d/%d] loss : %f" % (i, len(train_loader), epoch, epochs, loss))
    show_generated_data(X, fake_data)
