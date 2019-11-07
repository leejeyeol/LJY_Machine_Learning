import torch
import numpy as np
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import LJY_utils
import default_dir
from .model import *
#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--model', type=str, default='KD', help='Model name(for load and save)')
parser.add_argument('--dataset', type=str, default='MNIST', help='what is dataset?')
parser.add_argument('--outDir', default='', help="folder to output images and model checkpoints")

parser.add_argument('--netT', default='', help="path of pretrained networks.(explicitly. "
                                              "if it is ''(default) and pretrainedEpoch is not 0(default), "
                                              "default saveDir/'model'_'pretrainedEpoch' is path of pretrained model.")
parser.add_argument('--pretrainedEpoch', default=0, help="epoch of pretrained networks. 0(default) is on scratch learning.")

parser.add_argument('--is_cuda', action='store_true', help='enables cuda')
parser.add_argument('--is_save', action='store_true', help='enables save')
parser.add_argument('--is_display', action='store_true', help='enables display process in visdom')
parser.add_argument('--display_type', default='per_iter', help='disply per iter/epoch')

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')

parser.add_argument('--iteration', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')

parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nf', type=int, default=64, help='number of filters.')

parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)


dataDir, saveDir, resultDir = default_dir.dd_call(options.model+'_'+options.dataset)
LJY_utils.set_seed(options.seed)
LJY_utils.set_cuda(options.is_cuda)

dataloader = torch.utils.data.Dataloader(
    dset.MNIST(dataDir+'/MNIST',train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
teacher = netT()
netT.apply(LJY_utils.weights_init)
if options.netT != '':
    netT.load_state_dict(torch.load(options.netT))


loss_funciton = nn.CrossEntropyLoss()

optimizer = optim.adam(netT.parameters(),betas=(0.5,0.999),lr=2e-4)
print(1)