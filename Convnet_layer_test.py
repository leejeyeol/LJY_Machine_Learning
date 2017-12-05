import torch
import torch.nn as nn
from torch.autograd import Variable

# test input
batch_size = 20
in_channel = 3
width = 113
length = 113

# layers parameters
out_channel = [32, 64, 128, 256, 256, 200]
kernel_size = [5, 5, 5, 5, 5, 2]
stride = [2, 2, 2, 2, 2, 1]
padding = [1, 1, 1, 1, 1, 0]
layers = []


test_input = Variable(torch.FloatTensor(batch_size,in_channel,width,length))
print("test input")
print(test_input.size())
print("===============")

for i in range(len(out_channel)):
    layers.append(nn.Conv2d(in_channel, out_channel[i], kernel_size[i], stride[i], padding[i]))
    in_channel = out_channel[i]

for j in range(len(layers)):
    result = layers[j](test_input)
    print(result.size())
    test_input = result

print("===============")

# layers parameters
out_channel = [256, 256, 128, 64, 32, 1]
kernel_size = [2, 6, 5, 5, 6, 5]
stride = [1, 2, 2, 2, 2, 2]
padding = [0, 1, 1, 1, 1, 1]
layers = []

for i in range(len(out_channel)):
    layers.append(nn.ConvTranspose2d(in_channel, out_channel[i], kernel_size[i], stride[i], padding[i]))
    in_channel = out_channel[i]

for j in range(len(layers)):
    result = layers[j](test_input)
    print(result.size())
    test_input = result

