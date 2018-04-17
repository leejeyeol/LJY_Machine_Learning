import numpy as np



def mask_to_list(mask):

    list_form = [[mask.shape[0],mask.shape[1]]]
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y] == 1:
                list_form.append([x,y])
    return list_form


def list_to_mask(list_):
    mask = np.zeros((list_[0][0],list_[0][1],1))
    for i in range(1, len(list_)):
        mask[list_[i][0]][list_[i][1]] = 1
    return mask

