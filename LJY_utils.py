import os
import torch
from torch.autograd import Variable
import glob


# one hot generator
def one_hot(size, index):
    """ Creates a matrix of one hot vectors.
        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def make_dir(path):
    # if there is no directory, make a directory.
    # make_dir(save_path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("the save directory is maked.")
    return

def get_file_paths(path, separator, file_type):
    # return file list of the given type in the given path.
    # image_files = get_file_paths(image_folder, '/*.', ['png', 'PNG'])
    file_paths = []
    if not os.path.exists(path):
        return file_paths
    for extension in file_type:
        file_paths += glob.glob(path + separator + extension)
    file_paths.sort()
    return file_paths


def files_rename(files_path, extensions_list):
    file_list = get_file_paths(files_path, "/*.", extensions_list)

    # example : 1,11,111,112....12....2,21,211,212 ... => 000001,0000002...
    for i in range(0, len(file_list)):
        new_path = file_list[i].replace(file_list[i].split('/')[-1],'')+"%06d"%(int(file_list[i].split('/')[-1].split('.')[0]))+'.png'
        os.rename(file_list[i], new_path)

def extract_filename_from_path(path):
    name = path.split('/')[-1].split('.')[0]
    return name

