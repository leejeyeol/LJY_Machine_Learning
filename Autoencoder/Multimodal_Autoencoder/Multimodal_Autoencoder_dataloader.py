import os
import glob
import torch.utils.data
import numpy as np
from PIL import Image
import LJY_utils

class MMAE_Dataloader(torch.utils.data.Dataset):
    def __init__(self, RGB_path, Depth_path, Semantic_path, transfrom):
        super().__init__()

        assert os.path.exists(RGB_path)
        self.RGB_base_path = RGB_path

        assert os.path.exists(Depth_path)
        self.Depth_base_path = Depth_path

        #assert os.path.exists(Semantic_path)
        #self.Semantic_base_path = Semantic_path

        self.transform = transfrom

        RGB_paths = glob.glob(self.RGB_base_path + '/*.*')
        RGB_paths.sort()

        self.RGB_paths = RGB_paths

        Depth_paths = glob.glob(self.Depth_base_path + '/*.*')
        Depth_paths.sort()

        self.Depth_paths = Depth_paths
        '''
        Semantic_paths = glob.glob(self.Semantic_base_path + '/*.*')
        Semantic_paths.sort()

        self.Semantic_paths = Semantic_paths
        '''


    def __len__(self):
        return len(self.RGB_paths)

    def __getitem__(self, item):
        # RGB image Preprocessing ======================================================================================
        RGB_path = self.RGB_paths[item]
        img = self.pil_loader(RGB_path, 'RGB')
        RGB = self.transform(img)
        R = RGB[0].view((1, RGB.shape[1], RGB.shape[2]))
        G = RGB[1].view((1, RGB.shape[1], RGB.shape[2]))
        B = RGB[2].view((1, RGB.shape[1], RGB.shape[2]))

        #todo image tensor(3 w h) RGB => (1 w h) * 3  R,G,B

        Depth_path = self.Depth_paths[item]
        img = self.pil_loader(Depth_path, 'L')
        D = self.transform(img)
        D_mask = self.sparse_mask(D)

        '''
        # Semantic Preprocessing =======================================================================================
        # load semantic array(binary file from piecewisecrf (from https://github.com/Vaan5/piecewisecrf))
        with open(self.Semantic_paths[item] , 'rb') as array_file:
            ndim = np.fromfile(array_file, dtype=np.uint32, count=1)[0]
            shape = []
            for d in range(ndim):
                shape.append(np.fromfile(array_file, dtype=np.uint32, count=1)[0])
            array_data = np.fromfile(array_file, dtype=np.int16)
        S_ground, S_object, S_building, S_vegetation, S_sky = self.Semantic_to_binary_mask(np.reshape(array_data, shape))
        '''
        return R, G, B, D, D_mask #S_ground, S_object, S_building, S_vegetation, S_sky

    def pil_loader(self, path , convert = 'RGB'):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert(convert)

    def Semantic_to_binary_mask(self, semantic_array):
        ground = [2, 3]
        object = [4, 6, 7, 8, 9, 10]
        building = [1]
        vegetation = [5]
        sky = [0]
        semantic_list = [ground, object, building, vegetation, sky]
        result_list = [np.zeros(semantic_array.shape) for _ in range(5)]

        for i, semantic in enumerate(semantic_list):
            for type in semantic:
                mask = (semantic_array == type)
                masked = semantic_array * mask
                masked[masked != 0] = 1
                result_list[i] = result_list[i] + masked

        return result_list[0], result_list[1], result_list[2], result_list[3], result_list[4]
    def sparse_mask(self, sparse_matrix):
        return sparse_matrix != sparse_matrix.data.min()

class fold_MMAE_Dataloader(torch.utils.data.Dataset):
    def __init__(self, fold_number, fold_path,  transfrom, type = 'train'):
        super().__init__()
        self.type = type
        train_path, val_path = LJY_utils.fold_loader(fold_number, fold_path)
        if self.type =='train':
            self.RGB_paths = train_path[1]
            self.Depth_paths = train_path[0]
        #self.Semantic_base_path = Semantic_path
        elif self.type == 'validation':
            self.RGB_paths = val_path[1]
            self.Depth_paths = val_path[0]

        self.transform = transfrom

        # todo segmentation

    def __len__(self):
        return len(self.RGB_paths)

    def __getitem__(self, item):
        # RGB image Preprocessing ======================================================================================
        RGB_path = self.RGB_paths[item]
        img = self.pil_loader(RGB_path, 'RGB')
        RGB = self.transform(img)
        R = RGB[0].view((1, RGB.shape[1], RGB.shape[2]))
        G = RGB[1].view((1, RGB.shape[1], RGB.shape[2]))
        B = RGB[2].view((1, RGB.shape[1], RGB.shape[2]))

        # todo image tensor(3 w h) RGB => (1 w h) * 3  R,G,B

        Depth_path = self.Depth_paths[item]
        img = self.pil_loader(Depth_path, 'L')
        D = self.transform(img)
        D_mask = self.sparse_mask(D)

        '''
        # Semantic Preprocessing =======================================================================================
        # load semantic array(binary file from piecewisecrf (from https://github.com/Vaan5/piecewisecrf))
        with open(self.Semantic_paths[item] , 'rb') as array_file:
            ndim = np.fromfile(array_file, dtype=np.uint32, count=1)[0]
            shape = []
            for d in range(ndim):
                shape.append(np.fromfile(array_file, dtype=np.uint32, count=1)[0])
            array_data = np.fromfile(array_file, dtype=np.int16)
        S_ground, S_object, S_building, S_vegetation, S_sky = self.Semantic_to_binary_mask(np.reshape(array_data, shape))
        '''
        return R, G, B, D,D_mask  # S_ground, S_object, S_building, S_vegetation, S_sky

    def pil_loader(self, path, convert='RGB'):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert(convert)

    def Semantic_to_binary_mask(self, semantic_array):
        ground = [2, 3]
        object = [4, 6, 7, 8, 9, 10]
        building = [1]
        vegetation = [5]
        sky = [0]
        semantic_list = [ground, object, building, vegetation, sky]
        result_list = [np.zeros(semantic_array.shape) for _ in range(5)]

        for i, semantic in enumerate(semantic_list):
            for type in semantic:
                mask = (semantic_array == type)
                masked = semantic_array * mask
                masked[masked != 0] = 1
                result_list[i] = result_list[i] + masked

        return result_list[0], result_list[1], result_list[2], result_list[3], result_list[4]
    def sparse_mask(self, sparse_matrix):
        mask = sparse_matrix != sparse_matrix.min()
        return mask
