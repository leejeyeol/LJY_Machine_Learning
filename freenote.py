import glob as glob
import random
import os
import numpy as np

# sevaral data folder => 10(or any number) fold data paths.
# folder1     folder2          fold1.npy = [[folder1/data0, folder1/data22, ...],[folder2/label0, folder2/label22, ...]]
# -data0      -label0          fold2.npy = [[folder1/data2, folder1/data14, ...],[folder2/label2, folder2/label14, ...]]
# -data1      -label1    =>         ...
# -data2      -label2          foldn.npy = [[ ...],[ ...]]
# -data3      -label3
#    ...



path_roots_to_divide = ['/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/depth',
                   '/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train/img_ppm']
save_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/KITTI_train'
num_fold = 10
num_dataset = len(path_roots_to_divide)



len_paths = 0


all_paths = []
for root in path_roots_to_divide:
    paths = sorted(glob.glob(os.path.join(root,"*.*")))
    all_paths.append(paths)
    len_paths = len(paths)

rand_idx = [i for i in range(0, len_paths)]
random.shuffle(rand_idx)
num_data = int(len_paths/num_fold)

idx = 0
fold = [[[] for _ in range(num_dataset)] for _ in range(num_fold)]
for i in range(num_fold):
    for _ in range(num_data):
        for j in range(num_dataset):
           fold[i][j].append(all_paths[j][rand_idx[idx]])
        idx+=1

for i in range(num_fold):
    for j in range(num_fold):
        np.save(os.path.join(save_path, '%d_fold_divided_%d.npy' %(i,num_fold)), np.asarray(fold))



print(1)
