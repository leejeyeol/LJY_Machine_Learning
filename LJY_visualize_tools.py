import cv2
import numpy as np
def Test_Image(image, normalize = False):
    if normalize == True:
        image = (image-image.min())
        image = image / image.max()
    print("show image! please press any button.")
    cv2.imshow('result', image)
    cv2.waitKey()

