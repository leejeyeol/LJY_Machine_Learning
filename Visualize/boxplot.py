import numpy as np
import os
import glob
import matplotlib.pyplot as plt

data1=np.load(r'D:\experiments\dcgan_CelebA_timelog.npy')
data2=np.load(r'D:\experiments\ours_CelebA_timelog.npy')
data3=np.load(r'D:\experiments\alpha-gan_CelebA_timelog.npy')

data1=np.load(r'D:\experiments\MSSSIM\dcgan_30.npy')
data2=np.load(r'D:\experiments\MSSSIM\ours_20.npy')
data3=np.load(r'D:\experiments\MSSSIM\alpha-gan_30.npy')
data = np.stack((data1,data2,data3),1)
my_xticks = ['DCGAN', 'Ours(*)', 'alpha-GAN']

plt.boxplot(data, notch=False, patch_artist=False, showmeans=True, showfliers=False)
# plt.xlabel('methods')
plt.ylabel('---')
plt.xticks([i + 1 for i in range(len(my_xticks))], my_xticks)
plt.tight_layout()
plt.show()