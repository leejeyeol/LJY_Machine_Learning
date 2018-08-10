import numpy as np
import cv2
from matplotlib import pyplot as plt
stereo = cv2.StereoSGBM_create(0,64,11)

img_L = cv2.imread("/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI/2011_09_26_drive_0001_sync/image_02/data/0000000027.png", 0)
img_R = cv2.imread("/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI/2011_09_26_drive_0001_sync/image_03/data/0000000027.png",0)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
disparity = stereo.compute(img_L,img_R)

ax1.imshow(disparity, 'gray')
ax2.imshow(img_L,'gray')
plt.show()