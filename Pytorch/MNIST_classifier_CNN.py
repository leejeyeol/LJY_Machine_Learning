from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class MNIST_classifier_CNN(nn.Module):
  def __init__(self, class_num):
    super().__init__()
    self.class_num = class_num

    self.conv_net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
        nn.BatchNorm2d(10),
        nn.MaxPool2d(2),
        nn.ReLU(),

        nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
        nn.BatchNorm2d(20),
        nn.MaxPool2d(2),
        nn.ReLU()
    )
    self.fc_net = nn.Sequential(
        nn.Linear(320,50),
        nn.BatchNorm1d(50),
        nn.ReLU(),
        nn.Linear(50,self.class_num),
        nn.Softmax()
    )
  def forward(self, x):
    feature = self.conv_net(x)
    feature = feature.view(-1,320)
    y = self.fc_net(feature)
    return y


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


epochs = 5
learning_rate = 0.01
batch_size = 100
loss_function = nn.BCELoss()

# load the dataset
dataset = datasets.MNIST('../data', train=True,
                         download=True, transform=transforms.Compose([
        transforms.ToTensor()
        , transforms.Normalize((0.1307,), (0.3081,))
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
                       , transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

net = MNIST_classifier_CNN(class_num=10).cuda()  # gpu 사용.(뒤에 .cuda())
net.apply(weight_init)

optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999),
                       lr=learning_rate)  # Adam optimizer로 변경. betas =(0.5, 0.999)

train_loss_list = []
val_loss_list = []
net.train()
for epoch in range(epochs):
    for i, (X, t) in enumerate(train_loader):
        X = X.cuda()  # gpu 사용.(뒤에 .cuda()) => view를 이용해 vectorize하는 부분 사라짐
        t = one_hot_embedding(t, 10).cuda()  # gpu 사용.(뒤에 .cuda())

        Y = net(X)
        loss = loss_function(Y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation loss 계산. 계산이 무거우니 몇백 iteration혹은 몇 epoch마다 한번 수행하는것이 적당합니다. 예제는 매 100 iteration마다 수행합니다.
        if i % 100 == 0:
            with torch.no_grad():
                val_100_loss = []
                for (X, t) in valid_loader:
                    X = X.cuda()  # gpu 사용.(뒤에 .cuda())
                    t = one_hot_embedding(t, 10).cuda()  # gpu 사용.(뒤에 .cuda())

                    Y = net(X)
                    loss = loss_function(Y, t)
                    val_100_loss.append(loss)
                train_loss_list.append(loss)
                val_loss_list.append(np.asarray(val_100_loss).sum() / len(valid_loader))
        print("[%d/%d][%d/%d] loss : %f" % (i, len(train_loader), epoch, epochs, loss))

print("calculating accuracy...")
net.eval()
correct = 0
with torch.no_grad():
    for i, (X, t) in enumerate(test_loader):
        X = X.cuda()  # gpu 사용.(뒤에 .cuda())
        t = one_hot_embedding(t, 10).cuda()  # gpu 사용.(뒤에 .cuda())
        Y = net(X)

        onehot_y = softmax_to_one_hot(Y)
        correct += int(torch.sum(onehot_y * t))
print("Accuracy : %f" % (100. * correct / len(test_loader.dataset)))
plt.plot(np.column_stack((train_loss_list, val_loss_list)))
