import numpy as np
import os
import glob
import matplotlib.pyplot as plt
# time


'''
data1=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_time/gans.npy')
data2=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_time/ours.npy')
data3=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_time/ours+recon_learn.npy')
data4=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_time/ours+AAE.npy')
data5=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_time/base.npy')
data = np.stack((data1,data2,data3,data4,data5),1)
my_xticks = ['DCGAN', 'Ours(*)', 'Ours+Reconstruction', 'Ours_AAE', 'alpha-GAN']





'''

#MS-SSIM
ori_path = r'C:\Users\rnt\Desktop\VAEGAN_measure_results'
paths = sorted(glob.glob(os.path.join(ori_path, '*')))
data = []
label = []
for i in range(len(paths)):
    data.append(np.load(paths[i]))
    label.append(os.path.basename(paths[i]))

data = np.stack(data,1)
my_xticks = label



'''
# Wasserstein Critic
data1=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/gans_9.npy')
data2=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/ours_9.npy')
data3=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/bases_9.npy')
data = np.stack((data1,data2,data3),1)
my_xticks = ['DCGAN', 'Ours(*)', 'alpha-GAN']
'''
plt.boxplot(data, notch=False, patch_artist=False, showmeans= True,showfliers=False)
#plt.xlabel('methods')
plt.ylabel('PSNR')
plt.xticks([i+1 for i in range(len(my_xticks))], my_xticks)
plt.tight_layout()
plt.show()
#plt.savefig('/home/leejeyeol/Experiments/e_nose/test.eps')
