import numpy
import math
import cv2
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



if __name__ == '__main__':
  cases=  ['alpha-gan','alpha-gan','alpha-gan','alpha-gan','alpha-gan','ours','ours','ours','ours','ours']
  eps = [22,28,67,98,122,22,28,67,98,122]
  for i in range(len(cases)):
      supplementNaN = True
      case = cases[i]
      ep = eps[i]

      img_folder = os.path.join('/home/mlpa/Workspace/github/LJY_Machine_Learning/Pytorch/results', case+'_' +str(ep))
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

      np.save('~/data_ssd/workspace/experimental_result/LJY/VAEGAN_measure_results/%s_%d.npy'%(case,ep), np_PSNR_data)
      print('mean is %f' % result_mean)