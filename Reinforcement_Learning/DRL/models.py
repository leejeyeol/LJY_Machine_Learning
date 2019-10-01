import torch.nn as nn

class Cartpole_policy_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),

            nn.Linear(24, 24),
            nn.ReLU(),

            nn.Linear(24, 2),
            nn.Softmax()
        )
        print(self.main)

    def forward(self, input):
        output = self.main(input)
        return output

class Cartpole_policy_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),

            nn.Linear(24, 24),
            nn.ReLU(),

            nn.Linear(24, 2),
            nn.Softmax()
        )
        print(self.main)

    def forward(self, input):
        output = self.main(input)
        return output


