import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.optim as optim

device = torch.device("cuda:0")
criterion = nn.BCELoss()
epochs = 5
batch_size = 100
nz = 100
ngf = 64
ndf = 64


def show_generated_data(real_data, fake_data):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_data[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_data.detach()[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

# MNIST call and load
dataloader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Net_generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Net_generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf*4, kernel_size=5,stride=1,padding=1),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 5, 1, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 5, 2, 0),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 1, 4, 2, 0),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.generator(input)
        return output

net_generator = Net_generator(nz, ngf).to(device)
net_generator.apply(weights_init)

class Net_Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Net_Discriminator, self).__init__()
        self.ndf = ndf

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.ndf, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.ndf*4,1,3,1,0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.discriminator(input)
        return output

net_discriminator = Net_Discriminator(ndf).to(device)
net_discriminator.apply(weights_init)


#======================================================================================================================
# Training
#======================================================================================================================


test_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)

label = torch.FloatTensor(batch_size).to(device)
real_label = 1
fake_label = 0


# setup optimizer
optimizerD = optim.Adam(net_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_generator.parameters(), lr=0.001, betas=(0.5, 0.999))
G_losses = []
D_losses = []

for epoch in range(epochs):
    for i, (real_data, _) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        real_data = real_data.to(device)
        label.data.fill_(real_label)
        real_output = net_discriminator(real_data)
        errD_real = criterion(real_output, label)
        net_discriminator.zero_grad()
        errD_real.backward()

        # train with fake
        noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
        fake_data = net_generator(noise)
        label.data.fill_(fake_label)

        fake_output = net_discriminator(fake_data.detach())
        errD_fake = criterion(fake_output, label)
        errD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        net_generator.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        fake_output = net_discriminator(fake_data)
        errG = criterion(fake_output, label)
        errG.backward()
        optimizerG.step()


        G_losses.append(errG.item())
        D_losses.append(errD_real+errD_fake.item())

        if i %100 == 0 :
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f '
                  % (epoch, epochs, i, len(dataloader),
                     (errD_real + errD_fake).item(), errG.item()))
            show_generated_data(real_data,fake_data)


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


