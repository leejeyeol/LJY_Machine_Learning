import torch.nn as nn
import torch


class netT(nn.Module):
    def __init__(self, ngpu):
        super(netT, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 1*28*28 => 64*14*14
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200,10),
            nn.Softmax(10)

        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

