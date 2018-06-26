import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import glob
import os
import argparse
from numpy.linalg import inv
from numpy import linalg as LA
import enum
from skimage import io
import face_alignment

class Alignment(enum.Enum):
    TIGHT = 0
    LOOSE = 1
    NOT_ALIGNED = 2

class CNN(enum.Enum):
    TIGHT = 0
    LOOSE_SMALL = 1
    LOOSE_BIG = 2

# kCelebAEyeL = [68, 111]
# kCelebAEyeR = [107, 112]
# kCelebAWidth = 178
# kCelebAHeight = 218

kPathSize = [64, 224]
kEyeLooseL = np.float32([0.5-1/8, 1/2])
kEyeLooseR = np.float32([0.5+1/8, 1/2])
kEyeLooseDistance = LA.norm(kEyeLooseR - kEyeLooseL)
kEyeTightL = np.float32([20/64, 33/64])
kEyeTightR = np.float32([42/64, 33/64])
kEyeTightDistance = LA.norm(kEyeTightR - kEyeTightL)

kAlignMargin = 5/64
kAllowablePaddingRatio = 0.05

kResultFilePostfix = 'K-CYL2_김태훈'

# ======================================================================================================================
# Options
# ======================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory containing input test data')
parser.add_argument('--result_dir', type=str, help='save path')
options = parser.parse_args()
print(options)


def get_alignment_status(_img, _eyes_in_img):

    rows, cols, chs = _img.shape

    if rows != cols:
        return Alignment.NOT_ALIGNED

    tight_eye_L = rows * kEyeTightL
    tight_eye_R = rows * kEyeTightR

    if LA.norm(_eyes_in_img[0] - tight_eye_L) < kAlignMargin * rows \
            and LA.norm(_eyes_in_img[1] - tight_eye_R) < kAlignMargin * rows:
        return Alignment.TIGHT

    loose_eye_L = rows * kEyeLooseL
    loose_eye_R = rows * kEyeLooseR

    if LA.norm(_eyes_in_img[0] - loose_eye_L) < kAlignMargin * rows \
            and LA.norm(_eyes_in_img[1] - loose_eye_R) < kAlignMargin * rows:
        return Alignment.LOOSE

    return Alignment.NOT_ALIGNED


def get_similarity_transform_matrix(source_pts, target_pts):

    assert len(target_pts) == len(source_pts) > 1

    x1, y1 = source_pts[0]
    x2, y2 = source_pts[1]
    x1_, y1_ = target_pts[0]
    x2_, y2_ = target_pts[1]

    # calculate transform matrix
    T = np.float32([[x1, -y1, 1, 0], [y1, x1, 0, 1], [x2, -y2, 1, 0], [y2, x2, 0, 1]])
    X_ = np.float32([x1_, y1_, x2_, y2_])
    coefs = np.matmul(inv(T), X_)
    transform_mat = np.float32([[coefs[0], -coefs[1], coefs[2]], [coefs[1], coefs[0], coefs[3]]])

    # calculate error
    X1_ = np.matmul(transform_mat, np.float32([x1, y1, 1]))
    X2_ = np.matmul(transform_mat, np.float32([x2, y2, 1]))
    mse = 0.25 * (pow(X1_[0] - x1_, 2) + pow(X1_[1] - y1_, 2) + pow(X2_[0] - x2_, 2) + pow(X2_[1] - y2_, 2))
    print("MSE = %d" % mse)

    return transform_mat


def get_padding(_eyes_in_img, _aligned_eyes, _image_size, _target_size):

    # find corner points of returning image in input image
    ref_to_img = get_similarity_transform_matrix(_aligned_eyes, _eyes_in_img)
    ref_corners = np.float32([[0, 0, 1], [_target_size - 1, 0, 1], [_target_size - 1, _target_size - 1, 1],
                              [0, _target_size - 1, 1]])
    transformed_corners = np.matmul(ref_to_img, np.transpose(ref_corners))
    x_min, x_max = np.min(transformed_corners[0]), np.max(transformed_corners[0])
    y_min, y_max = np.min(transformed_corners[1]), np.max(transformed_corners[1])

    # calculate padding size
    pad_left = math.ceil(np.abs(min([x_min, 0])))
    pad_right = math.ceil(max([0, x_max - _image_size]))
    pad_top = math.ceil(np.abs(min([y_min, 0])))
    pad_bottom = math.ceil(max([0, y_max - _image_size]))

    return pad_left, pad_top, pad_right, pad_bottom


def get_aligned_image(_img, _eyes_in_img, _aligned_eyes, _target_size, pad_left, pad_top, pad_right, pad_bottom):

    img_pad = cv2.copyMakeBorder(_img, pad_top+2, pad_bottom+2, pad_left+2, pad_right+2, cv2.BORDER_REPLICATE)

    # shift to compensate padding
    _eyes_in_img += np.matlib.repmat(np.array([pad_left, pad_top]), len(_eyes_in_img), 1)

    # warp image
    img_to_ref = get_similarity_transform_matrix(_eyes_in_img, _aligned_eyes)
    img_aligned = cv2.warpAffine(img_pad, img_to_ref, (_target_size, _target_size))

    return img_aligned


def align_tight_face_image(_img, _eyes_in_img):
    rows, _, _ = _img.shape
    aligned_eyes = [kPathSize[0] * kEyeTightL, kPathSize[0] * kEyeTightR]
    pad_left, pad_top, pad_right, pad_bottom = get_padding(_eyes_in_img, aligned_eyes, rows, kPathSize[0])
    return get_aligned_image(_img, _eyes_in_img, aligned_eyes, kPathSize[0], pad_left, pad_top, pad_right,
                             pad_bottom), Alignment.TIGHT


def align_face_image(_img, _eyes_in_img):
    """
        Get transformed image on input(_img) with replicated padding.
        1. find corner points of returning image in input image for padding
        2. transform input(_img)

    :param _img: target image which will be warped
    :param _eyes_in_img: coordinates of left and right eye in the target image
    :return: aligned face image (insert padding with replicating), alignment type (loose or tight)
    """

    # =========================================================================
    # ALIGNMENT CHECK
    # =========================================================================

    alignment_res = get_alignment_status(_img, _eyes_in_img)
    rows, cols, chs = _img.shape

    if Alignment.NOT_ALIGNED != alignment_res:
        if Alignment.TIGHT == alignment_res or rows < kPathSize[1]:
            img_aligned = cv2.resize(_img, (kPathSize[0], kPathSize[0]))
        else:
            img_aligned = cv2.resize(_img, (kPathSize[1], kPathSize[1]))

        return img_aligned, alignment_res

    # =========================================================================
    # SELECT PATCH TYPE
    # =========================================================================
    # priority: big loose > small loose > tight
    eye_distance = LA.norm(np.float32(_eyes_in_img[0]) - np.float32(_eyes_in_img[1]))

    # try big loose
    for patch_size in kPathSize[::-1]:
        if eye_distance < (kEyeLooseDistance - kAlignMargin) * patch_size:
            continue

        aligned_eyes = [patch_size * kEyeLooseL, patch_size * kEyeLooseR]
        pad_left, pad_top, pad_right, pad_bottom = get_padding(_eyes_in_img, aligned_eyes, rows, patch_size)
        allowable_padding_size = kAllowablePaddingRatio * patch_size
        if pad_left > allowable_padding_size or pad_top > allowable_padding_size \
                or pad_right > allowable_padding_size or pad_bottom > allowable_padding_size:
            continue

        return get_aligned_image(_img, _eyes_in_img, aligned_eyes, patch_size, pad_left, pad_top, pad_right, pad_bottom), Alignment.LOOSE

    # try tight
    aligned_eyes = [kPathSize[0] * kEyeTightL, kPathSize[0] * kEyeTightR]
    pad_left, pad_top, pad_right, pad_bottom = get_padding(_eyes_in_img, aligned_eyes, rows, kPathSize[0])
    return get_aligned_image(_img, _eyes_in_img, aligned_eyes, kPathSize[0], pad_left, pad_top, pad_right, pad_bottom), Alignment.TIGHT


def save_result_to_txt(_result_dict_list, _file_path):
    with open(_file_path, "w") as res_file:
        for cur_dict in _result_dict_list:
            res_file.writelines(cur_dict['problem_no'] + ",%1.6f" % cur_dict['prob'])


def do_mission_1(_data_dir, _res_dir, _face_landmark_detector):
    # =========================================================================
    # MISSION 1
    # =========================================================================
    file_name_list = glob.glob(_data_dir + '/1_*.*')
    prediction_results = []
    for file_name in file_name_list:
        img = io.imread(file_name)
        cur_result = {'problem_no': file_name, 'prob': 1.0}  # <= default value is one to handle landmark missing

        # landmark prediction
        preds = _face_landmark_detector.get_landmarks(input)
        if preds is None:
            prediction_results.append(cur_result)
            continue

        # image alignment
        left_eye = [preds[36:42, 0].mean(), preds[36:42, 1].mean()]
        right_eye = [preds[42:48, 0].mean(), preds[42:48, 1].mean()]
        img_aligned, alignment_type = align_face_image(img, [left_eye, right_eye])

        if alignment_type == Alignment.TIGHT:
            print('tight')
            cur_result['prob'] = 1.0
            # todo Implement tight
        elif alignment_type == Alignment.LOOSE:
            print('loose')
            cur_result['prob'] = 1.0
            # todo Implement loose
            # if  shape is 64x64 = > LQ
            # elif shape is 224x224 => HQ

        prediction_results.append(cur_result)

    # save result
    save_result_to_txt(prediction_results, os.path.join(_res_dir, 'mission1_%s.txt' % kResultFilePostfix))


def do_mission_2(_data_dir, _res_dir, _face_landmark_detector):
    # =========================================================================
    # MISSION 2
    # =========================================================================
    file_name_list = glob.glob(_data_dir + '/2_*.*')
    prediction_results = []
    for file_name in file_name_list:
        img = io.imread(file_name)
        cur_result = {'problem_no': file_name, 'prob': 1.0}  # <= default value is one to handle landmark missing

        # do works
        # landmark prediction
        preds = _face_landmark_detector.get_landmarks(input)
        if preds is None:
            prediction_results.append(cur_result)
            continue

        prediction_results.append(cur_result)

    # save result
    save_result_to_txt(prediction_results, os.path.join(_res_dir, 'mission2_%s.txt' % kResultFilePostfix))


if __name__ == "__main__":

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=True)

    do_mission_1(options.data_dir, options.result_dir, fa)
    do_mission_2(options.data_dir, options.result_dir, fa)

# ()()
# ('') HAANJU & YEOLJERRY





