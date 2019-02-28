import os
import torch
from torch.autograd import Variable
import glob
import numpy as np
import math
import time
import csv
import random
class Deep_Learning_CSV_Saver():
    '''
    # Usage
    import random
    csv_saver = Deep_Learning_CSV_Saver(rows=['a', 'b', 'c', 'd'], save_path='output.csv')
    for i in range(0, 100):
        for j in range(0, 100):
            iteration_result = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]
            csv_saver.add_column(iteration_result)
        csv_saver.save()
    '''
    def __init__(self, rows=['1', '2', '3', '4'], save_path='output.csv'):
        self.results = []
        self.rows = rows
        self.len_rows = len(self.rows)
        self.save_path = save_path
        with open(self.save_path, 'a') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(self.rows)


    def add_column(self, data_list):
        assert len(data_list) == self.len_rows, 'length of input is not same to length of row. %d != %d' % (len(data_list), self.len_rows)
        self.results.append(data_list)

    def save(self):
        with open(self.save_path, 'a') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for item in self.results:
                # Write item to outcsv
                print(item)
                writer.writerow([item[0], item[1], item[2], item[3]])
            self.results = []


def torch_model_gradient(model_parameters):
    grad_list = []
    for f in model_parameters:
        num_of_weight = 1
        if type(f.grad) is not type(None):
            if torch.is_tensor(f.grad) is True:
                for i in range(len(f.grad.shape)):
                    num_of_weight *= f.grad.shape[i]
            else:
                for i in range(len(f.grad.data.shape)):
                    num_of_weight *= f.grad.shape[i]
            grad_list.append(float(torch.sum(torch.abs(f.grad))/num_of_weight))
    return np.asarray(grad_list).mean()


def measure_run_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print("'%s' function running time : %s"%(func.__name__, end - start))
        return result
    return wrapper
#@measure_run_time
# function

def Depth_evaluation(gt, predict, D_mask):
    height = gt.data[0].shape[1]
    width = gt.data[0].shape[2]
    num_of_pixels = height * width


    # average relative error
    rel = (torch.abs((gt-predict).data[0]))[0]
    sum = 0
    valid_pixels = 0
    for x in range(height):
        for y in range(width):
            if float(gt[0, 0, x, y].data) != 0:
                sum += rel[x, y]
                valid_pixels += 1
    rel = sum/valid_pixels


                # root mean squared error
    rms = math.sqrt((((gt-predict).pow(2).data[0]*D_mask.data[0].cuda().float()).sum()/num_of_pixels))

    return rel, rms
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
    elif classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.02)

#        nn.init.normal(m.weight.data)

def make_dir(path, allow_duplication = False):
    # if there is no directory, make a directory.
    # make_dir(save_path)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+" : the save directory is maked.")
    else:
        if allow_duplication:
            if not os.listdir(path):
                #empty. reuse it.
                path = path
                return path
            path = path + ':'
            path = make_dir(path, allow_duplication)
    return path

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
        new_path = file_list[i].replace(file_list[i].split('/')[-1],'')+"%06d"%(int(file_list[i].split('/')[-1].split('.')[0]))+extensions_list[0]
        os.rename(file_list[i], new_path)

def extract_filename_from_path(path):
    name = path.split('/')[-1].split('.')[0]
    return name

def integer_histogram(data):
    # data == array
    unique_elements = np.unique(data)
    histogram = np.histogram(data, (unique_elements.max()-unique_elements.min()))
    return histogram

def integer_histogram(data,min,max):
    histogram = np.histogram(data, bins=max-min, range=(min, max))
    return histogram

def three_channel_image_interger_histogram(data):
    unique_elements_0 = np.unique(data[:, 0])
    unique_elements_1 = np.unique(data[:, 1])
    unique_elements_2 = np.unique(data[:, 2])
    total_min = np.min([unique_elements_0.min(), unique_elements_1.min(), unique_elements_2.min()])
    total_max = np.max([unique_elements_0.max(), unique_elements_1.max(), unique_elements_2.max()])
    histogram_0 = np.histogram(data[:, 0], [x for x in range(total_min, total_max + 1)])
    histogram_1 = np.histogram(data[:, 1], [x for x in range(total_min, total_max + 1)])
    histogram_2 = np.histogram(data[:, 2], [x for x in range(total_min, total_max + 1)])
    histogram = [histogram_0[0], histogram_1[0], histogram_2[0]]

    return histogram

def three_channel_superpixel_interger_histogram_LAB(data):

    # mask =[weight, height, channel]
    histogram_0 = np.histogram(data[:, 0], [x for x in range(0, 100 + 1)])
    histogram_1 = np.histogram(data[:, 1], [x for x in range(-128, 127 + 1)])
    histogram_2 = np.histogram(data[:, 2], [x for x in range(-128, 127 + 1)])
    histogram = np.append(histogram_0[0], [histogram_1[0], histogram_2[0]])

    return histogram


def copy_2darray_into_3rd_dimension(array):
    return np.repeat(array[:, :, np.newaxis], 3, axis=2)

def HWC_to_CHW_3d(array):
    return np.transpose(array, (2, 0, 1))

class Time_calculator():
    def __init__(self):
        self.total_time_start()
        self.mean_time = []

    def total_time_start(self):
        self.total_start_time = time.time()

    def total_time_end(self):
        self.time_print(time.time()-self.total_start_time, "total spended time")

    def simple_time_start(self, string=None):
        self.simple_start_time = time.time()
        if string is None :
            self.simple_time_name = 'spended time'
        else:
            self.simple_time_name = string

    def simple_time_end(self):
        used_time = time.time() - self.simple_start_time
        self.mean_time.append(used_time)
        #self.time_print(used_time, self.simple_time_name)
        return used_time

    def mean_reset(self):
        self.mean_time = []
    def mean_calc(self):
        mean = np.asarray(self.mean_time).mean()
        self.time_print(mean, 'mean time per epoch')
        self.mean_reset()

    def time_print(self, tm, string):
        h = tm/360
        h_ = tm%360
        m = h_/60
        m_ = h_%60
        s = m_
        print("%s is %d:%02d:%.4f" % (string, h, m, s))


def fold_loader(fold_num, root_path):
    fold_num = int(fold_num)
    total_fold = int(os.path.basename(root_path).split('_')[-1])
    fold_list = glob.glob(os.path.join(root_path, "*.*"))

    train_paths = None
    for i in range(total_fold):
        if i != fold_num:
            if train_paths is None:
                train_paths = np.load(fold_list[i])
            else:
                train_paths = np.hstack((train_paths, np.load(fold_list[i])))

    val_paths = np.load(fold_list[fold_num])
    return train_paths, val_paths

def tvt_divider(cur_file_paths, train_ratio = 4, val_ratio = 1, test_ratio = 1, seed = 1111):
    #self.file_paths, self.val_paths, self.test_paths = LJY_utils.tvt_divider(cur_file_paths, train_ratio=4, val_ratio=1,test_ratio=1)
    random.seed(seed)
    random.shuffle(cur_file_paths)
    num_dataset = len(cur_file_paths)
    total_ratio = train_ratio + val_ratio + test_ratio

    train_idx = int(num_dataset / total_ratio * train_ratio)
    val_idx = train_idx + int(num_dataset / total_ratio * val_ratio)

    return cur_file_paths[: train_idx], cur_file_paths[train_idx: val_idx], cur_file_paths[val_idx:]

def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]