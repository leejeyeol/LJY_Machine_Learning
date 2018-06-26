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
import face_alignment
import Preprocessing.AI2018_face_alignment as AIFA


kTestPath = '/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_faceswap'
if __name__ == "__main__":
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=True)

    file_name_list = glob.glob(kTestPath+'/*.*')
    for file_name in file_name_list:
        img = io.imread(file_name)
        preds_allfaces = fa.get_landmarks(img, all_faces=True)

        multi_faces_result = []
        for preds_face in preds_allfaces:
            # align per face
            left_eye = [preds_face[36:42, 0].mean(), preds_face[36:42, 1].mean()]
            right_eye = [preds_face[42:48, 0].mean(), preds_face[42:48, 1].mean()]
            img_aligned, alignment_type = AIFA.align_face_image(img, [left_eye, right_eye])

            # todo : save image for test
            # todo : or feed to vgg16
            # todo : get result
            result_each_face = 0.4 # temporal value
            multi_faces_result.append(result_each_face)

        # decision
        if len([x for x in multi_faces_result if x > 0.5]) != 0:
            result = 1.0# todo calc probablity
        else:
            result = 0

