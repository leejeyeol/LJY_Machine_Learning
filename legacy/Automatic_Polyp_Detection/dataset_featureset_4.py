import os
import glob
import torch.utils.data
import numpy as np
import torch

class featureset_4(torch.utils.data.Dataset):
    def __init__(self, path, type= 'train'):
        super().__init__()
        self.add_string = lambda a, b: a + b


        assert os.path.exists(path)
        self.base_path = path


        # load feature.
        # [hist hog lm lbp]
        cur_file_paths = glob.glob(self.base_path + '/*.npy')
        cur_file_paths.sort()

        self.file_paths = cur_file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        data = np.load(self.file_paths[item])
        data[0] = data[0].astype(float)
        data[1] = data[1].astype(float)
        data[2] = data[2].astype(float)
        data[3] = data[3].astype(float)
        hist_feature = torch.FloatTensor(data[0])
        LM_feature = torch.FloatTensor(data[1])
        LBP_feature = torch.FloatTensor(data[2])
        HOG_feature = torch.FloatTensor(data[3])

        # todo data normalized
        return hist_feature, LM_feature, LBP_feature, HOG_feature, self.file_paths[item]


class featureset_4_online(torch.utils.data.Dataset):
    def __init__(self, superpixel):
        super().__init__()
        self.superpixel = superpixel

    def __len__(self):
        return len(self.superpixel)

    def __getitem__(self, item):
        sp = self.superpixel[item]

        hist_feature = sp.color_hist_feature.astype(float)
        HOG_feature= sp.HOG_feature.astype(float)
        LM_feature = sp.LM_feature.astype(float)
        LBP_feature = sp.LBP_feature.astype(float)

        hist_feature = torch.FloatTensor(hist_feature)
        LM_feature = torch.FloatTensor(LM_feature)
        LBP_feature = torch.FloatTensor(LBP_feature)
        HOG_feature = torch.FloatTensor(HOG_feature)

        # todo data normalized
        return hist_feature, LM_feature, LBP_feature, HOG_feature, sp.index