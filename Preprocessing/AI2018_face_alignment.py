import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from numpy.linalg import inv


kCelebAEyeL = [68, 111]
kCelebAEyeR = [107, 112]
kCelebAWidth = 178
kCelebAHeight = 218


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


def align_face_image(_img, _eyes_in_img, _aligned_eyes):
    """
        Get transformed image on input(_img) with replicated padding.
        1. find corner points of returning image in input image for padding
        2. transform input(_img)

    :param _img: target image which will be warped
    :param _eyes_in_img: coordinates of left and right eye in the target image
    :param _aligned_eyes: desired coordinates of left and right eye in the result
    :return: aligned face image (insert padding with replicating)
    """

    # =========================================================================
    # PADDING
    # =========================================================================

    # find corner points of returning image in input image
    ref_to_img = get_similarity_transform_matrix(_aligned_eyes, _eyes_in_img)
    ref_corners = np.float32([[0, 0, 1], [kCelebAWidth-1, 0, 1], [kCelebAWidth-1, kCelebAHeight-1, 1],
                              [0, kCelebAHeight-1, 1]])
    transformed_corners = np.matmul(ref_to_img, np.transpose(ref_corners))
    x_min, x_max = np.min(transformed_corners[0]), np.max(transformed_corners[0])
    y_min, y_max = np.min(transformed_corners[1]), np.max(transformed_corners[1])

    # make boarders
    rows, cols, chs = _img.shape
    pad_left = math.ceil(np.abs(min([x_min, 0]))) + 2
    pad_right = math.ceil(max([0, x_max - cols])) + 2
    pad_top = math.ceil(np.abs(min([y_min, 0]))) + 2
    pad_bottom = math.ceil(max([0, y_max - rows])) + 2
    img_pad = cv2.copyMakeBorder(_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    # =========================================================================
    # WARP
    # =========================================================================

    # shift to compensate padding
    _eyes_in_img += np.matlib.repmat(np.array([pad_left, pad_top]), len(_eyes_in_img), 1)

    # warp image
    img_to_ref = get_similarity_transform_matrix(_eyes_in_img, _aligned_eyes)
    img_aligned = cv2.warpAffine(img_pad, img_to_ref, (kCelebAWidth, kCelebAHeight))

    # visualization
    plt.subplot(121), plt.imshow(_img), plt.title('Input')
    plt.subplot(122), plt.imshow(img_aligned), plt.title('Output')
    # plt.show(block=False)
    plt.show()

    return img_aligned


if __name__ == "__main__":
    img = cv2.imread('1_00115.jpg')

    preds = np.load('1_00115.npy')

    left_eye = [preds[36:42, 0].mean(), preds[36:42, 1].mean()]
    right_eye = [preds[42:48, 0].mean(), preds[42:48, 1].mean()]

    align_face_image(img, [left_eye, right_eye], [kCelebAEyeL, kCelebAEyeR])