import os
import glob
import torch.utils.data
import numpy as np

class SandwichGAN_Dataloader(torch.utils.data.Dataset):
    #todo
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
        data = np.transpose(data, (0, 3, 1, 2))
        # (f h w c) => (f c h w)
        pre_frame, mid_frame, nxt_frame = data
        # (c h w)
        if pre_frame.dtype.name == 'uint8':
            pre_frame = pre_frame.astype(float)
        if mid_frame.dtype.name == 'uint8':
            mid_frame = mid_frame.astype(float)
        if nxt_frame.dtype.name == 'uint8':
            nxt_frame = nxt_frame.astype(float)

        if self.centered:
            pre_frame = torch.FloatTensor(pre_frame)
            mid_frame = torch.FloatTensor(mid_frame)
            nxt_frame = torch.FloatTensor(nxt_frame)
            return pre_frame, mid_frame, nxt_frame,"centered data"

        else:
            pre_min = pre_frame.min(axis=(1, 2), keepdims=True)
            pre_max = pre_frame.max(axis=(1, 2), keepdims=True)
            pre_frame = 2 * (pre_frame - pre_min) / (pre_max - pre_min) - 1

            mid_min = mid_frame.min(axis=(1, 2), keepdims=True)
            mid_max = mid_frame.max(axis=(1, 2), keepdims=True)
            mid_frame = 2 * (mid_frame - mid_min) / (mid_max - mid_min) - 1

            nxt_min = nxt_frame.min(axis=(1, 2), keepdims=True)
            nxt_max = nxt_frame.max(axis=(1, 2), keepdims=True)
            nxt_frame = 2 * (nxt_frame - nxt_min) / (nxt_max - nxt_min) - 1

            pre_frame = torch.FloatTensor(pre_frame)
            mid_frame = torch.FloatTensor(mid_frame)
            nxt_frame = torch.FloatTensor(nxt_frame)
            return pre_frame, mid_frame, nxt_frame, [pre_min, pre_max, mid_min, mid_max, nxt_min, nxt_max]



    #todo
    def get_decenterd_data(self, centered_data):
        result = centered_data.mul_(255) + self.mean_image
        result = result.byte()
        return result
    #todo
    def get_mean_image(self):
        mean_image = np.load(os.path.join(os.path.dirname(self.base_path), "mean_image.npy"))
        mean_image = np.transpose(mean_image, (2, 0, 1))
        mean_image = torch.from_numpy(mean_image).float()
        return mean_image