import numpy as np
import os
import glob
import matplotlib.pyplot as plt

data1_=np.load(r'D:\experiments\Timelog\dcgan_CelebA_timelog.npy')
data2_=np.load(r'D:\experiments\Timelog\ours_CelebA_timelog.npy')
data3_=np.load(r'D:\experiments\Timelog\alpha-gan_CelebA_timelog.npy')

data1=np.load(r'D:\experiments\MSSSIM\dcgan_30.npy')
data2=np.load(r'D:\experiments\MSSSIM\ours_30.npy')
data3=np.load(r'D:\experiments\MSSSIM\alpha-gan_30.npy')

data_ = np.stack((data1_,data2_,data3_),1)
data = np.stack((data1,data2,data3),1)

fig, ax1 = plt.subplots()
width = 0.6
x_blank = 0.4
my_xticks = ['DCGAN', 'Ours(*)', 'alpha-GAN']

color = 'tab:red'
box1=ax1.boxplot(data, notch=False, patch_artist=False, showmeans=True, showfliers=False, positions=np.array(range(len(my_xticks)))*2.0-x_blank, sym='', widths=width)
for box in box1['boxes']:
    # change outline color
    box.set(color=color, linewidth=2)
ax1.set_ylabel('MS-SSIM', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
box2 = ax2.boxplot(data_, notch=False, patch_artist=False, showcaps = False, showmeans=False, showfliers=False, positions=np.array(range(len(my_xticks)))*2.0+x_blank , sym='', widths=width)
for box in box2['boxes']:
    # change outline color
    box.set(color=color, linewidth=3)
ax2.set_ylabel('Time per iteration (min)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim(-0.3,0.7)
# plt.xlabel('methods')
#plt.ylabel('---')
plt.xlim(-1,5)
plt.xticks(range(0, len(my_xticks) * 2, 2), my_xticks)
plt.show()

'''

data1=np.load(r'D:\experiments\MSSSIM\dcgan_30.npy')
data2=np.load(r'D:\experiments\MSSSIM\ours_30.npy')


data = np.stack((data1,data2),1)

my_xticks = ['DCGAN', 'VLGAN(*)', 'alpha-GAN']
plt.boxplot(data, notch=False, patch_artist=False, showmeans=True, showfliers=False)
# plt.xlabel('methods')
plt.ylabel('---')
plt.xticks([i + 1 for i in range(len(my_xticks))], my_xticks)
plt.tight_layout()
plt.show()

'''