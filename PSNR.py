import numpy
import math
import cv2
import argparse

import glob
import os
import numpy as np
#original = cv2.imread("original.png")
#contrast = cv2.imread("photoshopped.png",1)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#d=psnr(original,contrast)

parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--preset', default='None', help='', choices=['None','ours','dcgan','alpha-gan'])
options = parser.parse_args()


if __name__ == '__main__':
    preset = options.preset

    #eps = [22,28,67,98,122,22,28,67,98,122]

    alpha_ep = [4, 5, 8, 10, 13, 15, 16, 19, 21, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 43, 44, 51,
              54, 55, 57, 59, 61, 66, 69, 72, 74, 75, 76, 78, 79, 82, 83, 85, 86, 87, 90, 92, 93, 94, 97, 99, 103, 104,
              105, 106, 107, 109, 110, 111, 112, 113, 118, 119, 120, 122, 131]

    ours_ep = [84, 13, 74, 57, 75, 57, 21, 54, 11, 16, 57, 106, 100, 97, 47, 57, 63, 47, 37, 123, 24, 57, 123, 28, 78, 64,
             93, 100, 73, 75, 40, 73, 23, 35, 34, 54, 123, 35, 24, 14, 24, 73, 43, 41, 77, 106, 118, 21, 106, 49, 47,
             43, 63, 22, 118, 123, 74, 21, 23, 13, 63, 22, 17, 84, 63, 15]

    cases = [preset for _ in range(len(preset))]

    for i in range(len(cases)):
        supplementNaN = True
        case = cases[i]
        if preset == 'alpha-gan':
            ep = alpha_ep[i]
        elif preset == 'ours':
            ep = ours_ep[i]


    img_folder = os.path.join('/home/mlpa/data_4T/experiment_results/ljy/results', case+'_' +str(ep))
    img_folder1 = os.path.join(img_folder, 'real')
    img_folder2 = os.path.join(img_folder, 'fake')
    imgs1 = sorted(glob.glob(os.path.join(img_folder1, "*")))
    imgs2 = sorted(glob.glob(os.path.join(img_folder2, "*")))
    num_of_imgs = len(imgs1) if len(imgs1) <= len(imgs2) else len(imgs2)

    result_data = []
    num_of_nan = 0
    for i in range(num_of_imgs):
        #print('%s,%s'%(imgs1[i],imgs2[i]))
        npImg1 = cv2.imread(imgs1[i])
        npImg2 = cv2.imread(imgs2[i])
        img1 = np.rollaxis(npImg1, 2).reshape((1,3,64,64))
        img2 = np.rollaxis(npImg2, 2).reshape((1,3,64,64))
        PSNR=(psnr(img1, img2))
        if not math.isnan(PSNR):
            print('[%d/%d] %f'%(i,num_of_imgs,PSNR))
            result_data.append(PSNR)
        else:
            print('nan')
            num_of_nan +=1

    result_data_ = np.asarray(result_data)
    result_mean = result_data_.mean()
    if supplementNaN :
        for i in range(num_of_nan):
            result_data.append(result_mean)

    np_PSNR_data = np.asarray(result_data)
    np.save('/home/mlpa/data_ssd/workspace/experimental_result/LJY/VAEGAN_measure_results/%s_%d.npy'%(case,ep), np_PSNR_data)
    print('mean is %f' % result_mean)