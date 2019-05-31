import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def convTranspose3x3(in_channels, out_channels, stride=1,padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample :
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Cifar10_resnet_D(nn.Module):
    def __init__(self, residualblock, layers, num_class=1):
        super(Cifar10_resnet_D, self).__init__()
        self.channels = 128
        self.conv = conv3x3(3, self.channels)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(residualblock, self.channels, layers[0])
        self.layer2 = self.make_layer(residualblock, self.channels, layers[1], 2)
        self.layer3 = self.make_layer(residualblock, self.channels, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.channels, num_class)

    def make_layer(self, residualblock, out_channels, num_of_residualblock, stride =1):
        downsample = None
        if (stride != 1) or (self.channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        maked_layers = []
        maked_layers.append(residualblock(self.channels, out_channels, stride, downsample))
        self.channels = out_channels
        for i in range(1, num_of_residualblock):
            maked_layers.append(residualblock(out_channels, out_channels))
        return nn.Sequential(*maked_layers)

    def forward(self,x):
        #print(x.shape)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.avg_pool(out)
        #print(out.shape)
        out = out.view(out.size(0), -1) # vectorize
        out = self.fc(out)
        #print(out.shape)
        return  out

class Cifar10_resnet_E(nn.Module):
    def __init__(self, residualblock, layers, nz = 64, type = 'AE'):
        super(Cifar10_resnet_E, self).__init__()
        self.channels = 128
        self.type = type
        self.nz = nz

        self.conv = conv3x3(3, self.channels)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(residualblock, self.channels, layers[0])
        self.layer2 = self.make_layer(residualblock, self.channels, layers[1], 2)
        self.layer3 = self.make_layer(residualblock, self.channels, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.channels, nz)
        self.fc_mu = nn.Linear(nz, nz)
        self.fc_sig = nn.Linear(nz, nz)

    def make_layer(self, residualblock, out_channels, num_of_residualblock, stride =1):
        downsample = None
        if (stride != 1) or (self.channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        maked_layers = []
        maked_layers.append(residualblock(self.channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_of_residualblock):
            maked_layers.append(residualblock(out_channels, out_channels))
        return nn.Sequential(*maked_layers)

    def forward(self,x):
        if len(x.shape) == 2:
            x = x.view(x.shape[0],x.shape[1],1,1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.avg_pool(out)
        #print(out.shape)
        out = out.view(out.size(0), -1) # vectorize
        out = self.fc(out)
        if self.type == 'VAE':
            mu = self.fc_mu(out)
            sig = self.fc_sig(out)
            return mu, sig
        return  out

class ResidualBlock_Reverse(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 1):
        super(ResidualBlock_Reverse, self).__init__()
        self.conv1 = convTranspose3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convTranspose3x3(out_channels,out_channels,stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        # if self.upsample :
        #    residual = self.upsample(x)
        return out

class Cifar10_resnet_G(nn.Module):
    def __init__(self, residualblock, layers, nz=16):
        super(Cifar10_resnet_G, self).__init__()
        self.channels = 128
        self.fcn = nn.Linear(nz, self.channels*4*4)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)


        self.layer1 = self.make_layer(residualblock, self.channels, layers[0], 2, self.upsample)
        self.layer2 = self.make_layer(residualblock, self.channels, layers[1], 2, self.upsample)
        self.layer3 = self.make_layer(residualblock, self.channels, layers[2], 2, self.upsample)

        self.conv = conv3x3(self.channels,3)
        self.tanh = nn.Tanh()

    def make_layer(self, residualblock, out_channels, num_of_residualblock, stride, upsample):

        maked_layers = []
        #maked_layers.append(residualblock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, num_of_residualblock):
            maked_layers.append(residualblock(out_channels, out_channels))
        maked_layers.append(upsample)
        return nn.Sequential(*maked_layers)

    def forward(self,x):
        x = x.view(x.size(0), x.size(1))
        out = self.fcn(x)
        out = out.view(out.size(0), self.channels, 4, 4)
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.conv(out)
        out = self.tanh(out)
        #print(out.shape)

        return  out



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    batch_size = 16
    nz = 128
    train_dataset = torchvision.datasets.CIFAR10(root=r'C:\Users\rnt\Desktop\V',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size = batch_size, shuffle=True)

    model_D = Cifar10_resnet_D(ResidualBlock, [2,2,2]).to(device)
    model_G = Cifar10_resnet_G(ResidualBlock_Reverse, [2,2,2], nz).to(device)
    model_E = Cifar10_resnet_E(ResidualBlock, [2,2,2],nz,'VAE').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_D.parameters(), lr=0.0002)

    for i ,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model_D(images)

        #print('generator')

        fixed_noise_128 = torch.randn(batch_size, nz,1,1)
        fixed_noise_128 = fixed_noise_128.to(device)
        noisev = fixed_noise_128
        samples = model_G(noisev)

        #print('encoder')
        mu, sig = model_E(images)



        print(1)
    print(1)

