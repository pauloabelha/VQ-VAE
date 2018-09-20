import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import io_image
import numpy as np



class YCB_Dataset(Dataset):

    root_folder = ''
    length = 0

    colour_filepaths = {}
    depth_filepaths = {}
    num_types_of_file = 3 # colour, mask and depth

    def __init__(self, root_folder, img_res=(64, 64)):
        self.img_res = img_res
        self.root_folder = root_folder
        self.file_idxs = set([])
        num_files = 0
        for root, dirs, files in os.walk(root_folder, topdown=True):
            for filename in sorted(files):
                if 'colour' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.file_idxs |= set([file_idx])
                    self.colour_filepaths[file_idx] = root + filename
                    num_files += 1
                if 'depth' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.depth_filepaths[file_idx] = root + filename
        self.file_idxs = list(self.file_idxs)
        self.length = len(self.file_idxs)


    def __getitem__(self, idx):
        colour = io_image.read_RGB_image(
            self.colour_filepaths[self.file_idxs[idx]],
            new_res=self.img_res)
        depth = io_image.read_RGB_image(
            self.depth_filepaths[self.file_idxs[idx]],
            new_res=self.img_res)
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
        RGBD_image = np.concatenate((colour, depth), axis=-1).astype(float)
        RGBD_image = np.divide(RGBD_image, 255)
        RGBD_image = RGBD_image.swapaxes(1, 2).swapaxes(0, 1)
        RGBD_image = torch.from_numpy(RGBD_image).float()
        return RGBD_image, 0

    def __len__(self):
        return self.length



def DataLoader(root_folder, batch_size=4, img_res=(64, 64)):
    dataset = YCB_Dataset(root_folder, img_res=img_res)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)