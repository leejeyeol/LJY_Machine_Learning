import os
import cv2
import LJY_utils
import LJY_visualize_tools
import numpy as np


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Ground_Truth")

#=======================================================================================================================
#   Functions
#=======================================================================================================================
def png_to_npy(original_image_path):
    # call imagepath list
    save_path = os.path.join(folder_path, "Ground_Truth")
    LJY_utils.make_dir(save_path)
    image_list=LJY_utils.get_file_paths(original_image_path, "/*.", ['png', 'PNG'])
    # load image and ..

    for (i, image) in enumerate(image_list):
        original_image = cv2.imread(image, -1)
        np.save(os.path.join(save_path, (LJY_utils.extract_filename_from_path(image))), original_image)



#=======================================================================================================================
#   run
#=======================================================================================================================

png_to_npy(original_image_path)