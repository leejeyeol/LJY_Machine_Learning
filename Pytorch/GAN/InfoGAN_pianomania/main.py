# base : https://github.com/pianomania/infoGAN-pytorch/blob/master/main.py
from Pytorch.GAN.InfoGAN_pianomania.model import *
from Pytorch.GAN.InfoGAN_pianomania.trainer import Trainer

nz = 60
nc = 4

fe = FrontEnd()
d = D()
q = Q(nc)
g = G()
e = E(nz, 'VAE')
for i in [fe, d, q, g, e]:
  i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q, e, nz, nc)
trainer.train()