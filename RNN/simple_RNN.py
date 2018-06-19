import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTM(1, 64)
        self.lstm2 = nn.LSTM(64, 1)
        self.p = 0.5

    def forward(self, seq, hc=None):
        out = []
        if hc == None:
            hc1, hc2 = None, None
        else:
            hc1, hc2 = hc
        X_in = torch.unsqueeze(seq[0], 0)
        for X in seq.chunk(seq.size(0), dim=0):
            if np.random.rand() > self.p:
                X_in = X
            tmp, hc1 = self.lstm1(X_in, hc1)
            X_in, hc2 = self.lstm2(tmp, hc2)
            out.append(X_in)
        return torch.stack(out).squeeze(1), (hc1, hc2)

seq = Sequence()
criterion = nn.MSELoss()
optimizer = optim.Adam(seq.parameters(), lr=0.001)
for i in range(1000):
    data = np.sin(np.linspace(0,10,100)+2*np.pi*np.random.rand())
    xs = data[:-1]
    ys = data[1:]
    X = Variable(torch.Tensor(xs).view(-1,1,1))
    y = Variable(torch.Tensor(ys))
    if i%100==0:
        seq.p = min(seq.p+0.1,0.85)  # encourage training longer term predictions
    optimizer.zero_grad()
    lstm_out,_ = seq(X)
    loss = criterion(lstm_out[20:].view(-1),y[20:])
    loss.backward()
    optimizer.step()
    if i%10 == 0:
        print("i {}, loss {}".format(i,loss.data.numpy()[0]))