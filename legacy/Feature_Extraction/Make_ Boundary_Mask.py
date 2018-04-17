import cv2
import numpy as np
import os
import LJY_utils
import LJY_visualize_tools

#todo : remove test code
test_code = False


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Original")
save_path = os.path.join(folder_path, "boundary_mask")
LJY_utils.make_dir(save_path)

def Make_Boundary_Mask():
    image_list = LJY_utils.get_file_paths(original_image_path, "/*.", ['png', 'PNG'])

    # load image and convert to Gray and convert to uint8 (original paper)
    initial_image, image_shape=Image_Load_and_Convert_to_uint8Gray(image_list[0])

    boundary_mask = np.zeros((len(image_list), initial_image.shape[0], initial_image.shape[1]))
    for x in range(0, image_shape[0]):
        for y in range(0, image_shape[1]):
            if initial_image[x, y] > 30:
                # find 'black point' per pixel.
                boundary_mask[0, x, y] = 255
                # it's true, set its value to 255.

    for (i, image) in enumerate(image_list):
        original_image, _ = Image_Load_and_Convert_to_uint8Gray(image)
        for x in range(0, image_shape[0]):
            for y in range(0, image_shape[1]):
                if original_image[x, y] > 30:
                    # find 'black point' per pixel.
                    boundary_mask[i, x, y] = 255
        print('[%d/%d] %s' % (i+1, len(image_list), image))
        if test_code:
            idx = boundary_mask[0,:,:] == 0
            LJY_visualize_tools.Test_Image(boundary_mask[i])
        else:
            cv2.imwrite(os.path.join(save_path, "boundary_mask_%06d.png")%(i+1), boundary_mask[i])

def Image_Load_and_Convert_to_uint8Gray(image):
    _image = cv2.imread(image, 0)
    _shape = _image.shape
    _image = cv2.convertScaleAbs(_image)
    return _image, _shape

Make_Boundary_Mask()
    # call image list
    # find [r<12,G<12,B<12]: black boundary. set 255 on it and 0 on otherwise
    # iteration , add, and calc mean.
