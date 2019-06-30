import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import LJY_utils

class RL_GAN():
    def __init__(self):
        class State_generator(nn.Module):
            def __init__(self, nz):
                super().__init__()
                self.main = nn.Sequential(
                nn.Linear(nz, 24),
                nn.ReLU,

                nn.Linear(24, 24),
                nn.ReLU,

                nn.Linear(24,4)
                )
            def forward(self, *input):
                fake_state = self.main(input)
                return fake_state
        self.state_generator = State_generator(2)
        self.state_generator.apply(LJY_utils.weights_init)
        self.state_generator.cuda()

        class Env_approximator(nn.Module):
            def __init__(self, n_action, n_state):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(n_action + n_state , 24),
                    nn.ReLU,

                    nn.Linear(24, 24),
                    nn.ReLU,

                    nn.Linear(24, 24),
                    nn.ReLU
                )
                self.reward_fc = nn.Linear(24, 1)
                self.nxt_state_fc = nn.Linear(24, 4)

            def forward(self, *input):
                out = self.main(input)
                return self.reward_fc(out), self.nxt_state_fc(out)

        self.env_approximator = Env_approximator(1, 4)
        self.env_approximator.apply(LJY_utils.weights_init)
        self.env_approximator.cuda()

        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.Linear(10, 24),
                    nn.ReLU,

                    nn.Linear(24, 24),
                    nn.ReLU,

                    nn.Linear(24, 1)
                )

            def forward(self, *input):
                fake_state = self.main(input)
                return fake_state

        self.discriminator = Discriminator()
        self.discriminator.apply(LJY_utils.weights_init)
        self.discriminator.cuda()
        self.loss = nn.BCELoss().cuda()
        self.optimizerG = optim.Adam(self.state_generator.parameters(), betas=(0.5, 0.999), lr=2e-4)
        self.optimizerG_env = optim.Adam(self.env_approximator.parameters(), betas=(0.5, 0.999), lr=2e-4)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), betas=(0.5, 0.999), lr=2e-4)

    def generate_state(self, train=False):
        noise = torch.Tensr(1, 2)
        noise.data.normal_(0,1)
        if train :
            fake_state = self.state_generator(noise)
        else :
            with torch.no_grad():
                fake_state = self.state_generator(noise)
        return fake_state






if __name__ == '__main__':
    print(1)