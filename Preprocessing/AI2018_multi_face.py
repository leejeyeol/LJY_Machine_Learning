import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import glob
import os
from numpy.linalg import inv
from numpy import linalg as LA
import enum
from skimage import io
from skimage import color
import face_alignment
import Preprocessing.AI2018_face_alignment as AIFA
from PIL import Image

kTestPath = '/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_faceswap/original'
kSavePath = '/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_faceswap/aligned'

tttt ='/home/leejeyeol/Downloads/BEGAN_generated_sample'
if __name__ == "__main__":
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=True)

    tlist = glob.glob(tttt+'/*.*')

    left_eyes = []
    right_eyes = []
    for i,t in enumerate(tlist):
        img = io.imread(t)
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        preds = fa.get_landmarks(img)
        left_eye = [preds[0][36:42, 0].mean(), preds[0][36:42, 1].mean()]
        right_eye = [preds[0][42:48, 0].mean(), preds[0][42:48, 1].mean()]
        left_eyes.append(left_eye)
        right_eyes.append(right_eye)
        print(i)
        if i == 50:
            break

    print(12313)





'''

    file_name_list = glob.glob(kTestPath+'/*.*')
    i = 0
    for file_name in file_name_list:
        img = io.imread(file_name)
        if img.shape[2] == 4:
            img = color.rgba2rgb(img)
        print(os.path.basename(file_name))
        preds_allfaces = fa.get_landmarks(img, all_faces=True)

        if preds_allfaces is None:
            result = 1.0
        else:
            multi_faces_result = []
            for preds_face in preds_allfaces:
                # align per face
                left_eye = [preds_face[36:42, 0].mean(), preds_face[36:42, 1].mean()]
                right_eye = [preds_face[42:48, 0].mean(), preds_face[42:48, 1].mean()]
                img_aligned, alignment_type = AIFA.align_loose_image(img, [left_eye, right_eye])

                # todo : save image for test
                Image.fromarray(img_aligned).save('%s/%06d.png'%(kSavePath, i))
                i+=1
                # todo : or feed to vgg16
                # todo : get result
                result_each_face = 0.4 # temporal value
                multi_faces_result.append(result_each_face)

            # decision
            if len([x for x in multi_faces_result if x > 0.5]) != 0:
                result = 1.0# todo calc probablity
            else:
                result = 0
        print('%s is %f'%(os.path.basename(file_name), result))
'''
