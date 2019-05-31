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
'''
# Wasserstein Critic
data1=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/gans_9.npy')
data2=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/ours_9.npy')
data3=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/VAEGAN_WC/bases_9.npy')
data = np.stack((data1,data2,data3),1)
my_xticks = ['DCGAN', 'Ours(*)', 'alpha-GAN']


paths = sorted(glob.glob(os.path.join(ori_path, '*')))
data = []
label = []
for i in range(len(paths)):
    data.append(np.load(paths[i]))
    label.append(os.path.basename(paths[i]))

data = np.stack(data,1)
my_xticks = label
'''
alpha_ep = [4, 5, 8, 10, 13, 15, 16, 19, 21, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 43, 44, 51,
            54, 55, 57, 59, 61, 66, 69, 72, 74, 75, 76, 78, 79, 82, 83, 85, 86, 87, 90, 92, 93, 94, 97, 99, 103, 104,
            105, 106, 107, 109, 110, 111, 112, 113, 118, 119, 120, 131]

ours_ep = [84, 13, 74, 57, 75, 57, 21, 54, 11, 16, 57, 106, 97, 47, 57, 63, 47, 37, 123, 24, 57, 123, 28, 78, 64,
           93, 100, 73, 75, 40, 73, 23, 35, 34, 54, 123, 35, 24, 14, 24, 73, 43, 41, 77, 106, 118, 21, 106, 49, 47,
           43, 63, 22, 118, 123, 74, 21, 23, 13, 63, 22, 17, 84, 15]

#MS-SSIM
ori_path = r'C:\Users\rnt\Desktop\VAEGAN_measure_results'
data = []
label = []
print(len(alpha_ep))
for i in range(len(alpha_ep)):

    alpha_data=np.load(os.path.join(ori_path, 'alpha-gan_%d.npy' % alpha_ep[i]))
    data.append(alpha_data)
    label.append('%d_alpha-gan'%alpha_ep[i])
    if alpha_data.size != 10000:
        print(alpha_ep[i])
        print(ours_ep[i])
        print(i)
    ours_data = np.load(os.path.join(ori_path,'ours_%d.npy'%ours_ep[i]))
    data.append(ours_data)
    label.append('%d_ours' % ours_ep[i])
    if ours_data.size != 10000:
        print(alpha_ep[i])
        print(ours_ep[i])
        print(i)
    if len(data) == 8 :
        data = np.stack(data, 1)
        my_xticks = label

        plt.boxplot(data, notch=False, patch_artist=False, showmeans=True, showfliers=False)
        # plt.xlabel('methods')
        plt.ylabel('PSNR')
        plt.xticks([i + 1 for i in range(len(my_xticks))], my_xticks)
        plt.tight_layout()
        plt.show()
        # plt.savefig('/home/leejeyeol/Experiments/e_nose/test.eps')

        label.append('%d_alpha_gan'%alpha_ep[i])
        label.append('%d_ours'%ours_ep[i])
        data = []
        label = []

