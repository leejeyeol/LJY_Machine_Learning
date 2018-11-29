'''from torch.autograd import Variable
import pytorch_msssim

import torch
import glob as glob
import numpy as np
import cv2
import os
import torch.nn.functional as f

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_msssim.MSSSIM()

#img1 = torch.rand(1, 1, 256, 256)
#img2 = torch.rand(1, 1, 256, 256)


img_folder1 = '/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/generated/1000recon_real_sample'
img_folder2 = '/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/generated/1000recon_generated'
imgs1 = sorted(glob.glob(os.path.join(img_folder1,"*")))
imgs2 = sorted(glob.glob(os.path.join(img_folder2,"*")))
num_of_imgs = len(imgs1) if len(imgs1)<=len(imgs2) else len(imgs2)

for i in range(num_of_imgs):
    npImg1 = cv2.imread(imgs1[i])
    npImg2 = cv2.imread(imgs2[i])

    img1 = Variable(torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0).cuda()
    img2 = Variable(torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0).cuda()

    print(float(pytorch_msssim.msssim(img1, img2)))
    print("%0.3f    %0.3f    %0.3f    %0.3f"%(img1.data.min(),img1.data.max(),img2.data.min(),img2.data.max()))

    #print(m(img1, img2))
'''

import numpy as np

import matplotlib.pyplot as plt
data1=np.load('/media/leejeyeol/74B8D3C8B8D38750/Experiment/plot/data/real-generated.npy')

#data = np.stack((data1,data2,data3,data4,data5),1)

plt.boxplot(data1, notch=False, patch_artist=False)
#plt.xlabel('methods')
plt.ylabel('time per iteration (%)')
#my_xticks = ['DCGAN','Ours(*)','Ours+Reconstruction','Ours_AAE','alpha-GAN']
#plt.xticks([1, 2, 3, 4,5],my_xticks)
plt.tight_layout()
plt.show()
#plt.savefig('/home/leejeyeol/Experiments/e_nose/test.eps')
