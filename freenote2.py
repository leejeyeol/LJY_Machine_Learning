
import pytorch_msssim
import torch
from torch.autograd import Variable


m = pytorch_msssim.MSSSIM()

img1 = Variable(torch.rand(1, 1, 256, 256)).cuda()
img2 = Variable(torch.rand(1, 1, 256, 256)).cuda()

print(float((pytorch_msssim.msssim(img1, img2))))
print(m(img1, img2))
