import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

class TwoLayerNet_pytorch(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.network1 = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.Sigmoid(),
        nn.Linear(self.hidden_size, self.output_size),
        nn.Sigmoid()
    )
  def forward(self, x):
    y = self.network1(x)
    return y

def weight_init(m):
    classname = m.__class__.__name__
    # m에서 classname이 Linear(신경망 레이어)인 경우
    if classname.find('Linear') != -1:
        # weight를 uniform distribution을 이용하여 초기화하고 bias는 0으로 초기화
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class xor_dataloader(torch.utils.data.Dataset):
    # 신경망 학습 중 신경망에 데이터를 공급해주는 dataloader 정의
    def __init__(self):
        super().__init__()
        # 외부 파일을 이용할 경우 여기서 파일을 읽거나 파일들의 경로를 찾음
        self.input = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.target = torch.Tensor([[0], [1], [1], [0]])

    def __len__(self):
        # dataloder로 기능하기위해 선언 필요.
        # 데이터의 총 개수가 출력되도록 함
        return len(self.input)

    def __getitem__(self, item):
        # dataloader에 데이터를 요청하였을 때 다음 데이터를 제공.
        X = self.input[item]
        t = self.target[item]
        return X, t

epochs = 20000
net = TwoLayerNet_pytorch(input_size=2, hidden_size=3, output_size=1)
net.apply(weight_init)
optimizer = optim.SGD(net.parameters(), lr=0.1)
dataloader = torch.utils.data.DataLoader(xor_dataloader(), batch_size=4)
loss_function = nn.MSELoss()
train_loss_list = []
for epoch in range(epochs):
    for i, (X, t) in enumerate(dataloader,0):
        Y = net(X)
        loss = loss_function(Y, t)
        train_loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# 학습종료 후 결과물 확인 (optional)
print("Input is")
print(X)
print("expected output is")
print(t)
print("actual output is ")
print(Y)
