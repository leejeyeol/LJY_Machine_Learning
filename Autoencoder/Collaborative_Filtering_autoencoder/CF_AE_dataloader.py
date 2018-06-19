import os
import glob
import torch.utils.data
import numpy as np

class CF_AE_Dataloader(torch.utils.data.Dataset):
    #todo
    def __init__(self, path):
        super().__init__()

        assert os.path.exists(path)
        self.data_path = path
        user_item_matrix = np.load(self.data_path)

        self.user_item_matrix = user_item_matrix/5
        print("debug")


    def __len__(self):
        return self.user_item_matrix.shape[0]

    def __getitem__(self, item):
        data = self.user_item_matrix[item]
        data = torch.FloatTensor(data)

        return data
