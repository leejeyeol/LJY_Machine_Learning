import cv2
import numpy as np
import matplotlib.pyplot as plt

kCelebAEyeL = [68, 111]
kCelebAEyeR = [107, 112]
kCelebAWidth = 178
kCelebAHeight = 218

def align_face_image(_img, _eyes_in_img, _aligned_eyes):

    rows, cols, ch = _img.shape

    plt.imshow(img), plt.title('Input')
    plt.plot(*zip(*_eyes_in_img))
    plt.plot(*zip(*_aligned_eyes))
    plt.show()

    pts1 = np.float32(_eyes_in_img)
    pts2 = np.float32(_aligned_eyes)
    template_to_image = cv2.estimateRigidTransform(pts1, pts2, False)

    dst = cv2.warpAffine(_img, template_to_image, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


if __name__ == "__main__":
    img = cv2.imread('1_00115.jpg')

    preds = np.load('1_00115.npy')

    left_eye = [preds[36:42, 0].mean(), preds[36:42, 1].mean()]
    right_eye = [preds[42:48, 0].mean(), preds[42:48, 1].mean()]

    align_face_image(img, [left_eye, right_eye], [kCelebAEyeL, kCelebAEyeR])