import os
import cv2
import LJY_utils
import LJY_visualize_tools
import numpy as np


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Reflection Removed")
sample_size = (224, 224)


#=======================================================================================================================
#   Functions
#=======================================================================================================================
def Resize_and_remove_boundary(original_image_path):
    threshold_of_mask = 30
    save_path = os.path.join(folder_path, "Remove_Boundary")
    LJY_utils.make_dir(save_path)
    # call imagepath list
    image_list=LJY_utils.get_file_paths(original_image_path, "/*.", ['png', 'PNG'])
    # load image and ..

    for (i, image) in enumerate(image_list):
        original_image = cv2.imread(image, 0)

        mask = np.ones([original_image.shape[0], original_image.shape[1]])


        for x in range(0,original_image.shape[0]):
            #print(np.var(original_image[x,:]))

            if np.max(original_image[x,:]) < threshold_of_mask:
                mask[x, :] = 0

        for y in range(0, original_image.shape[1]):
            #print(np.var(original_image[:, y]))

            if np.max(original_image[:, y]) < threshold_of_mask:
                mask[:, y] = 0



        original_image = cv2.imread(image, -1)
        ground_truth = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(image)),"Ground_Truth",os.path.split(image)[1]), -1)


        mask_x_start, mask_x_end, mask_y_start, mask_y_end = find_xy_start_and_end_of_rantangle_mask(mask)
        crop_image = original_image[mask_x_start:mask_x_end, mask_y_start:mask_y_end, :]
        crop_image = cv2.resize(crop_image, sample_size)
        cv2.imwrite(os.path.join(save_path, os.path.split(image)[-1]), crop_image)

        crop_gt = ground_truth[mask_x_start:mask_x_end, mask_y_start:mask_y_end]
        crop_gt = cv2.resize(crop_gt, sample_size)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(image)),"Ground_Truth",os.path.split(image)[1]), crop_gt)

        print("[%d/%d]" % (i+1, len(image_list)))


def find_xy_start_and_end_of_rantangle_mask(mask):
    find_end = False
    find_start = False

    for x in range(mask.shape[0]-1, 0, -1):
        if find_end:
            break
        for y in range(mask.shape[1]-1, 0, -1):
            if mask[x, y] == 1:
                x_end = x
                y_start = y
                find_end = True

    for x in range(0, mask.shape[0]):
        if find_start:
            break
        for y in range(0, mask.shape[1]):
            if mask[x, y] == 1:
                x_start = x
                y_end = y
                find_start = True

    return x_start, x_end, y_start, y_end
#=======================================================================================================================
#   run
#=======================================================================================================================

Resize_and_remove_boundary(original_image_path)