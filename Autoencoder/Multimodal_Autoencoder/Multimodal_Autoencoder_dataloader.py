import os
import glob
import torch.utils.data
import numpy as np

class MMAE_Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, centered=False):
        super().__init__()
        self.centered = centered
        self.add_string = lambda a, b: a + b

        assert os.path.exists(path)
        self.base_path = path

        #self.mean_image = self.get_mean_image()

        cur_file_paths = glob.glob(self.base_path + '/*.npy')
        cur_file_paths.sort()
        self.file_paths = cur_file_paths


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        data = np.load(self.file_paths[item])
        R, G, B, D= data
        # (c h w)
        if R.dtype.name == 'uint8':
            R = R.astype(float)
        if G.dtype.name == 'uint8':
            G = G.astype(float)
        if B.dtype.name == 'uint8':
            B = B.astype(float)
        if D.dtype.name == 'uint8':
            D = D.astype(float)

        if self.centered:
            R = torch.FloatTensor(R)
            G = torch.FloatTensor(G)
            B = torch.FloatTensor(B)
            D = torch.FloatTensor(D)
            #return R.view(-1), G.view(-1), B.view(-1), D.view(-1)
            return R.view(1,18,60), G.view(1,18,60), B.view(1,18,60), D.view(1,18,60)
        else:
            R = 2 * (R) / (255) - 1
            G = 2 * (G) / (255) - 1
            B = 2 * (B) / (255) - 1
            D = 2 * (D) / (255) - 1
            #D = D - D.mean()
            R = torch.FloatTensor(R)
            G = torch.FloatTensor(G)
            B = torch.FloatTensor(B)
            D = torch.FloatTensor(D)
            #return R.view(-1), G.view(-1), B.view(-1), D.view(-1)
            return R.view(1,18,60),G.view(1,18,60),B.view(1,18,60),D.view(1,18,60)

