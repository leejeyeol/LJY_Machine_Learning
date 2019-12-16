import glob as glob
import random
import os
import numpy as np

# train - validation - test set divider

data_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png'
save_path = data_path
data_paths = glob.glob(os.path.join(data_path, "*"))
num_dataset = len(data_paths)
train_ratio = 4
val_ratio = 1
test_ratio = 1
total_ratio = train_ratio + val_ratio + test_ratio

train_idx = int(num_dataset/total_ratio*train_ratio)
val_idx = train_idx+int(num_dataset/total_ratio*val_ratio)

random.seed(1111)
random.shuffle(data_paths)

train_data = data_paths[: train_idx]
val_data = data_paths[train_idx: val_idx]
test_data = data_paths[val_idx:]


np.save(os.path.join(save_path, '%train_paths.npy'), np.asarray(train_data))
np.save(os.path.join(save_path, '%val_paths.npy'), np.asarray(val_data))
np.save(os.path.join(save_path, '%test_paths.npy'), np.asarray(test_data))


