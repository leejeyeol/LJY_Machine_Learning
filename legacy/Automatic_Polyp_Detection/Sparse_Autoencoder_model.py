import torch
import torch.nn as nn
from torch.autograd import Function

class L1Penalty(Function):
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input

class SAE(nn.Module):
    def __init__(self, ngpu, feature_size, z_size=10):
        super().__init__()
        self.ngpu = ngpu
        self.l1weight = 5
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Linear(feature_size, z_size),
            nn.Sigmoid(),

            # state size: (nf) x 113 x 113
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_size, feature_size),
            nn.Sigmoid()
        )

        # init weights
        self.weight_init()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        z = L1Penalty.apply(z, self.l1weight)
        return self.decode(z), z

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.decoder.apply(weight_init)

# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        print("xavier")
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)