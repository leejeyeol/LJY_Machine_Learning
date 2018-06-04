import numpy as np

a= np.load('/home/leejeyeol/Git/LJY_Machine_Learning/GAN/AI2018/output/analysis.npy')

print(a.mean(axis=0))

print(a.std(axis=0))
