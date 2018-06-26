import os
import glob
import torch.utils.data
import numpy as np

class SandwichGAN_Dataloader(torch.utils.data.Dataset):
    #todo
    def __init__(self, path, transform, centered=False):
        super().__init__()
        self.transform = transform
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
        pre_frame, mid_frame, nxt_frame = data

        pre_frame = self.transform(pre_frame)
        mid_frame = self.transform(mid_frame)
        nxt_frame = self.transform(nxt_frame)

        return pre_frame, mid_frame, nxt_frame



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