import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from skimage.transform import resize


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=True)
#input = io.imread('/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png/000417.png') no face

left_eyes = []
right_eyes = []
error_image = []

eye_mean = [0 for i in range(0,400)]
for i in range(1,400):
    #input = io.imread('/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png/%06d.png'%i)
    input = io.imread('/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_test/1_%05d.jpg'%i)
    #input = io.imread('/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_test/1_00369.jpg')

    faces = fa.detect_faces(input)

    if len(faces) == 1:
        preds = fa.get_landmarks(input)
        if preds is None:
            error_image.append(i)
            break
        preds = preds[-1]
        left_eyes.append([preds[36:42, 0].mean(), preds[36:42, 1].mean()])
        right_eyes.append([preds[42:48, 0].mean(), preds[42:48, 1].mean()])
        np.save('/media/leejeyeol/74B8D3C8B8D38750/second_exp/AI2018/landmarks/1_%05d.npy' % i, preds)
        print("%d" % i)

    elif len(faces) == 0:
        resized_input = resize(input, (input.shape[0] * 4, input.shape[0] * 4))
        faces = fa.detect_faces(resized_input)
        if len(faces) == 0:
            error_image.append(i)
            print('no landmarks %d' % i)
        else:
            preds = fa.get_landmarks(resized_input)
            if preds is None:
                error_image.append(i)
                break
            preds = preds[-1]

            left_eye_mean = ([preds[36:42, 0].mean(), preds[36:42, 1].mean()])
            right_eye_mean = ([preds[42:48, 0].mean(), preds[42:48, 1].mean()])
            np.save('/media/leejeyeol/74B8D3C8B8D38750/second_exp/AI2018/landmarks/1_%05d.npy' % i, preds)
            print("%d" % i)

    else:
        for i_face, face in enumerate(faces):
            left = face.left()
            right = face.right()
            top = face.top()
            bottom = face.bottom()
            face_image = input[top:bottom, left:right, :]

            preds = fa.get_landmarks(face_image)
            if preds is None:
                error_image.append(i)
                break
            preds = preds[-1]
            left_eyes.append([preds[36:42,0].mean(),preds[36:42,1].mean()])
            right_eyes.append([preds[42:48,0].mean(), preds[42:48, 1].mean()])

            print("%d"%i)

print("left eye coordi mean : %d,%d" % (np.asarray(left_eyes).mean(0)[0], np.asarray(left_eyes).mean(0)[1]))
print("left eye coordi var : %d,%d" % (np.asarray(left_eyes).var(0)[0], np.asarray(left_eyes).var(0)[1]))

print("right eye coordi mean : %d,%d" % (np.asarray(right_eyes).mean(0)[0], np.asarray(right_eyes).mean(0)[1]))
print("right eye coordi var : %d,%d" % (np.asarray(right_eyes).var(0)[0], np.asarray(right_eyes).var(0)[1]))

print("can't find face list : ")
print(error_image)