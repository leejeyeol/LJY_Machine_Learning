import os
import cv2
import LJY_utils
import LJY_visualize_tools
import numpy as np


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Original")

#=======================================================================================================================
#   Functions
#=======================================================================================================================
def Specular_Reflection_Detection(original_image_path):
    save_path = os.path.join(folder_path, "Reflection Removed")
    LJY_utils.make_dir(save_path)
    # call imagepath list
    image_list=LJY_utils.get_file_paths(original_image_path, "/*.", ['png', 'PNG'])
    # load image and ..

    for (i, image) in enumerate(image_list):
        original_image = cv2.imread(image, -1)


        specular_reflection = Find_Specular_Reflection(original_image)
        specular_reflection = cv2.dilate(specular_reflection, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        reflection_removed_image = Repainting_Specular_Reflection(specular_reflection, original_image)
        cv2.imwrite(os.path.join(save_path, os.path.split(image)[-1]), reflection_removed_image)
        print("[%d/%d]"%(i+1, len(image_list)))


    # find Specular reflection
    # remove specular reflection
    # save clean image to save path

def Find_Specular_Reflection(image, saturation_threshold = 0.29*255, intensity_threshold = 0.65*255):
    # rgb to hsi
    hsi_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    specular_map = cv2.inRange(hsi_image, np.array([0,0,intensity_threshold]), np.array([255,saturation_threshold,255]))
    return specular_map

def Repainting_Specular_Reflection(specular_reflection, original_image):
    # find specular point in the image 255
    tmp_origianl_image = original_image
    tmp_specular_reflection = specular_reflection
    while(np.max(tmp_specular_reflection)):
        for x in range(0, tmp_specular_reflection.shape[0]):
            for y in range(0, tmp_specular_reflection.shape[1]):
                if tmp_specular_reflection[x][y] == 255:
                    #print("x: %d        y: %d"%(x,y))
                    tmp_specular_reflection, tmp_origianl_image = check_8_neighbhors(x, y, tmp_specular_reflection, tmp_origianl_image)
    return tmp_origianl_image


def check_8_neighbhors(x,y,specular_reflection, original_image):
    # (3,3) size window
    # counter clockwise
    tmp_specular_reflection = specular_reflection
    tmp_original_image = original_image
    if (x-2) < 0 or (y-2) < 0 or (x+2) > tmp_specular_reflection.shape[0] or (y+2) > tmp_specular_reflection.shape[1]:
        # detect abnormally case
        print("Specular Reflection of Corner is detected.")
    else:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not(i == 0 and j == 0):
                    # only when not in the center.
                    tmp_specular_reflection, tmp_original_image = Calc_Window(x, y, i, j, tmp_specular_reflection, tmp_original_image)
    return tmp_specular_reflection, tmp_original_image

def Calc_Window(x, y, i, j, specular_reflection, original_image):
    # calculation per window
    tmp_original_image = original_image
    tmp_specular_reflection = specular_reflection
    counter = 0
    total_of_values = np.zeros(3)
    for k in range(x+i-1, x+i+2):
        for l in range(y+j-1, y+j+2):
            if tmp_specular_reflection[k][l] == 0 and not (tmp_original_image[k][l][0]==0 and tmp_original_image[k][l][1]==0 and tmp_original_image[k][l][2]==0):
                counter = counter + 1
                total_of_values[0] = total_of_values[0] + tmp_original_image[k][l][0]
                total_of_values[1] = total_of_values[1] + tmp_original_image[k][l][1]
                total_of_values[2] = total_of_values[2] + tmp_original_image[k][l][2]
    if counter > 6 and tmp_specular_reflection[x][y] == 255:
        #print("yes")
        total_of_values[0] = int(total_of_values[0]/counter)
        total_of_values[1] = int(total_of_values[1]/counter)
        total_of_values[2] = int(total_of_values[2]/counter)
        #print(total_of_values)
        tmp_original_image[x][y] = total_of_values
        tmp_specular_reflection[x][y] = 0
        return tmp_specular_reflection, tmp_original_image
    else:
        #print("no. counter : %d"%counter)
        return specular_reflection, original_image




#=======================================================================================================================
#   run
#=======================================================================================================================

Specular_Reflection_Detection(original_image_path)