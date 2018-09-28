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
                vis.image(image_list[i], win=win_dict["%d" % i])
        else:
            for i, name in enumerate(name_list):
                vis.image(image_list[i], win=win_dict[name])
    return win_dict

def draw_lines_to_windict(win_dict, value_list, legend_list, epoch, iteration, total_iter):
    # epoch * total_iter + iteration

    num_of_values = len(value_list)
    value_list = np.asarray(value_list)

    if type(win_dict) == dict:
        # first. line plots
        if legend_list is None :
            win_dict = vis.line(X=np.column_stack((0 for _ in range(num_of_values))),
                                Y=np.column_stack((value_list[i] for i in range(num_of_values))),
                                opts=dict(
                               title='loss-iteration',
                               xlabel='iteration',
                               ylabel='loss',
                               xtype='linear',
                               ytype='linear',
                               makers=False
                           ))
        else:
            win_dict = vis.line(X=np.column_stack((0 for _ in range(num_of_values))),
                                Y=np.column_stack((value_list[i] for i in range(num_of_values))),
                                opts=dict(
                                    legend=legend_list,
                                    title='loss-iteration',
                                    xlabel='iteration',
                                    ylabel='loss',
                                    xtype='linear',
                                    ytype='linear',
                                    makers=False
                                ))

    else:
        win_dict = vis.line(
            X=np.column_stack((epoch*total_iter + iteration for _ in range(num_of_values))),
            Y=np.column_stack((value_list[i] for i in range(num_of_values))),
            win=win_dict,
            update='append'
        )
    return win_dict

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.

        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        unorm(tensor)
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor