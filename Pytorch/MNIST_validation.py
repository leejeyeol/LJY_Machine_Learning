import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def one_hot_embedding(labels, num_classes):
    # 단일 라벨 텐서를 원핫 벡터로 바꿔줍니다.
    y = torch.eye(num_classes)
    one_hot = y[labels]
    return one_hot

def softmax_to_one_hot(tensor):
    # softmax 결과를 가장 높은 값이 1이 되도록 하여 원핫 벡터로 바꿔줍니다. acuuracy 구할 때 씁니다.
    max_idx = torch.argmax(tensor, 1, keepdim=True)
    if tensor.is_cuda :
        one_hot = torch.zeros(tensor.shape).cuda()
    else:
        one_hot = torch.zeros(tensor.shape)
    one_hot.scatter_(1, max_idx, 1)
    return one_hot

def weight_init(m):
    classname = m.__class__.__name__
    # m에서 classname이 Linear(신경망 레이어)인 경우
    if classname.find('Linear') != -1:
        # weight를 uniform distribution을 이용하여 초기화하고 bias는 0으로 초기화
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


class TwoLayerNet_pytorch(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.network1 = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.BatchNorm1d(self.hidden_size), # batch normalization 추가
        nn.ReLU(), # ReLU로 교체
        nn.Linear(self.hidden_size, self.output_size),
        nn.Softmax()
    )
  def forward(self, x):
    y = self.network1(x)
    return y

epochs = 1
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
valid_size = 100

indices = list(range(num_train))
split = num_train-valid_size
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
                       ,transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

net = TwoLayerNet_pytorch(input_size=784, hidden_size=50, output_size=10).cuda() # gpu 사용.(뒤에 .cuda())
net.apply(weight_init)

optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=learning_rate)  # Adam optimizer로 변경. betas =(0.5, 0.999)

train_loss_list = []
val_loss_list = []
net.train()
for epoch in range(epochs):
    for i, (X, t) in enumerate(train_loader):
        X = X.view(-1, 784).cuda() # gpu 사용.(뒤에 .cuda())
        t = one_hot_embedding(t, 10).cuda() # gpu 사용.(뒤에 .cuda())

        Y = net(X)
        loss = loss_function(Y, t)
        train_loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_100_loss = []
            for (X, t) in valid_loader:
              X = X.view(-1, 784).cuda() # gpu 사용.(뒤에 .cuda())
              t = one_hot_embedding(t, 10).cuda() # gpu 사용.(뒤에 .cuda())

              Y = net(X)
              loss = loss_function(Y, t)
              val_100_loss.append(loss)
            val_loss_list.append(np.asarray(val_100_loss).sum()/len(valid_loader))
        print("[%d/%d][%d/%d] loss : %f"%(i,len(train_loader),epoch,epochs, loss))

print("calculating accuracy...")
net.eval()
correct = 0
with torch.no_grad():
    for i, (X, t) in enumerate(test_loader):
        X = X.view(-1, 784).cuda() # gpu 사용.(뒤에 .cuda())
        t = one_hot_embedding(t, 10).cuda() # gpu 사용.(뒤에 .cuda())
        Y = net(X)

        onehot_y= softmax_to_one_hot(Y)
        correct += int(torch.sum(onehot_y * t))
print("Accuracy : %f" % (100. * correct / len(test_loader.dataset)))
plt.plot([train_loss_list,val_loss_list])