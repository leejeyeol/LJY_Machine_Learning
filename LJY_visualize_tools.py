import cv2
from visdom import Visdom
import torch
import numpy as np
from torch.autograd import Variable

vis = Visdom()

def Test_Image(image, normalize = False):
    if normalize == True:
        image = (image-image.min())
        image = image / image.max()
    print("show image! please press any button.")
    cv2.imshow('result', image)
    cv2.waitKey()

def Tensor_list_to_npy(tensor_list):
    output_list = []
    for tensor in tensor_list:
        if isinstance(tensor, Variable):
            tensor = tensor.data
        else:
            tensor = tensor
        output_list.append(tensor.cpu().numpy())
    return output_list

def win_dict():
    win_dict = dict(
        exist=False)
    return win_dict

def draw_images_to_windict(win_dict, image_list, name_list = None):
    image_list = Tensor_list_to_npy(image_list)
    if not win_dict['exist']:
        win_dict['exist'] = True
        if name_list is None:
            for i in len(image_list):
                win_dict['%d' % i] = vis.image(image_list[i], opts=dict(title="%d" % i))
        else:
            for i, name in enumerate(name_list):
                win_dict[name] = vis.image(image_list[i], opts=dict(title=name))
    else:
        if name_list is None:
            for i in len(image_list):
                vis.image(image_list[i], win=win_dict["%d" % i]
)
        else:
            for i, name in enumerate(name_list):
                vis.image(image_list[i], win=win_dict[name]
)
    return win_dict
