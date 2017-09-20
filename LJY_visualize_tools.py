import cv2

def Test_Image(image):
    cv2.imshow('result', image)
    print("show image! please press any button.")
    cv2.waitKey()
