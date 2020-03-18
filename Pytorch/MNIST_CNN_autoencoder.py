from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class MNIST_CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class MNIST_CNN_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x_ = self.decoder(z)
        return x_




def one_hot_embedding(labels, num_classes):
    # 단일 라벨 텐서를 원핫 벡터로 바꿔줍니다.
    y = torch.eye(num_classes)
    one_hot = y[labels]
    return one_hot


def softmax_to_one_hot(tensor):
    # softmax 결과를 가장 높은 값이 1이 되도록 하여 원핫 벡터로 바꿔줍니다. acuuracy 구할 때 씁니다.
    max_idx = torch.argmax(tensor, 1, keepdim=True)
    if tensor.is_cuda:
        one_hot = torch.zeros(tensor.shape).cuda()
    else:
        one_hot = torch.zeros(tensor.shape)
    one_hot.scatter_(1, max_idx, 1)
    return one_hot


def weight_init(m):
    # Conv layer와 batchnorm layer를 위한 가중치 초기화를 추가함.
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


epochs = 10
learning_rate = 0.01
batch_size = 100
loss_function = nn.MSELoss()

# load the dataset
dataset = datasets.MNIST('../data', train=True,
                         download=True, transform=transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize((0.5,), (0.5,))
    ]))
num_train = len(dataset)
valid_size = 500

indices = list(range(num_train))
split = num_train - valid_size
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                       , transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size, shuffle=True)

encoder = MNIST_CNN_Encoder().cuda()
encoder.apply(weight_init)

decoder = MNIST_CNN_Decoder().cuda()
decoder.apply(weight_init)

net_params = list(encoder.parameters())+list(decoder.parameters())
optimizer = optim.Adam(net_params, betas=(0.5, 0.999),lr=learning_rate)

train_loss_list = []
val_loss_list = []
encoder.train()
decoder.train()
for epoch in range(epochs):
    for i, (X, _) in enumerate(train_loader):
        X = X.cuda()
        z = encoder(X)
        recon_X = decoder(z)

        loss = loss_function(recon_X, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation loss 계산.
        if i % 100 == 0:
            with torch.no_grad():
                val_100_loss = []
                for (X, _) in valid_loader:
                    X = X.cuda()
                    z = encoder(X)
                    recon_X = decoder(z)
                    loss = loss_function(recon_X, X)

                    val_100_loss.append(loss)
                train_loss_list.append(loss)
                val_loss_list.append(np.asarray(val_100_loss).sum() / len(valid_loader))
        print("[%d/%d][%d/%d] loss : %f" % (i, len(train_loader), epoch, epochs, loss))

print("testing")
encoder.eval()
decoder.eval()
correct = 0
with torch.no_grad():
    for i, (X, _) in enumerate(test_loader):
        X = X.cuda()
        z = encoder(X)
        recon_X = decoder(z)

        print("테스트 결과")
        for i in range(5):
            plt.imshow(X[i].cpu().reshape(28, 28))
            plt.gray()
            plt.show()

            plt.imshow(recon_X[i].cpu().reshape(28, 28))
            plt.gray()
            plt.show()
        break

plt.plot(np.column_stack((train_loss_list, val_loss_list)))

#################################


class MNIST_CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

class MNIST_FCN(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

        self.fc_net = nn.Sequential(
            nn.Linear(32, 50),
            nn.ReLU(),
            nn.Linear(50, self.class_num),
            nn.Softmax()
        )
    def forward(self, x):
        x = x.view(-1, 32)
        y = self.fc_net(x)
        return y

fcn = MNIST_FCN(class_num=10).cuda()
fcn.apply(weight_init)

pretrained_encoder = MNIST_CNN_Encoder().cuda()
saved_weights = encoder.state_dict()
pretrained_encoder.load_state_dict(saved_weights)
#pretrained_encoder.apply(weight_init)

epochs = 5
learning_rate = 0.01
batch_size = 100
loss_function = nn.BCELoss()

optimizer = optim.Adam(list(fcn.parameters())+list(pretrained_encoder.parameters()), betas=(0.5, 0.999), lr=learning_rate)
#optimizer = optim.Adam(fcn.parameters(), betas=(0.5, 0.999), lr=learning_rate)  # Adam optimizer로 변경. betas =(0.5, 0.999)

train_loss_list = []
fcn.train()
for epoch in range(epochs):
    for i, (X, t) in enumerate(train_loader):
        X = X.cuda()
        t = one_hot_embedding(t, 10).cuda()
        z = pretrained_encoder(X)
        Y = fcn(z)

        loss = loss_function(Y, t)
        train_loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("[%d/%d][%d/%d] loss : %f"%(i,len(train_loader),epoch,epochs, loss))

print("calculating accuracy...")
fcn.eval()
correct = 0
with torch.no_grad():
    for i, (X, t) in enumerate(test_loader):
        X = X.cuda()
        t = one_hot_embedding(t, 10).cuda()
        z = pretrained_encoder(X)
        Y = fcn(z)

        onehot_y= softmax_to_one_hot(Y)
        correct += int(torch.sum(onehot_y * t))
print("Accuracy : %f" % (100. * correct / len(test_loader.dataset)))
plt.plot(train_loss_list)