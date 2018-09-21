import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import io_image
import numpy as np
import visualize as vis
from numpy import genfromtxt
import util


class YCB_Dataset(Dataset):

    root_folder = ''
    length = 0
    transform = None

    colour_filepaths = {}
    depth_filepaths = {}
    mask_filepaths = {}
    with_imagenet_filepaths = {}
    pose_filepaths = {}

    num_types_of_file = 3 # colour, mask and depth

    def __init__(self, root_folder, transform=None, img_res=(64, 64), num_channels=4):
        self.transform = transform
        self.img_res = img_res
        self.root_folder = root_folder
        self.file_idxs = set([])
        self.num_channels = num_channels
        num_files = 0
        for root, dirs, files in os.walk(root_folder, topdown=True):
            for filename in sorted(files):
                if 'pose' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.file_idxs |= set([file_idx])
                    self.pose_filepaths[file_idx] = root + filename
                elif 'imagenet' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.file_idxs |= set([file_idx])
                    self.with_imagenet_filepaths[file_idx] = root + filename
                elif 'colour' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.file_idxs |= set([file_idx])
                    self.colour_filepaths[file_idx] = root + filename
                    num_files += 1
                elif 'depth' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.depth_filepaths[file_idx] = root + filename
                elif 'mask' in filename:
                    file_idx = int(filename.split('_')[-1].split('.')[0]) - 1
                    self.mask_filepaths[file_idx] = root + filename
        self.file_idxs = list(self.file_idxs)
        self.length = len(self.file_idxs)

    def crop_image(self, image, mask):
        data_crop_nonzero = np.nonzero(mask)
        crop_min = np.array([np.min(data_crop_nonzero[0]), np.min(data_crop_nonzero[1])])
        crop_max = np.array([np.max(data_crop_nonzero[0]), np.max(data_crop_nonzero[1])])
        cropped_img = image[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], :]
        return cropped_img

    def __getitem__(self, idx):
        pose =  genfromtxt(self.pose_filepaths[self.file_idxs[idx]], delimiter=',')[1:]
        pose = torch.from_numpy(pose).float()

        colour = io_image.read_RGB_image(self.colour_filepaths[self.file_idxs[idx]])
        mask = io_image.read_RGB_image(self.mask_filepaths[self.file_idxs[idx]])
        cropped_img = self.crop_image(colour, mask)
        colour = io_image.change_res_image(colour, self.img_res)
        mask = io_image.change_res_image(mask, self.img_res)
        cropped_img = io_image.change_res_image(cropped_img, self.img_res)

        with_imagenet = io_image.read_RGB_image(
            self.with_imagenet_filepaths[self.file_idxs[idx]],
            new_res=self.img_res)

        data_image = with_imagenet
        if self.num_channels > 3:
            depth = io_image.read_RGB_image(
                self.depth_filepaths[self.file_idxs[idx]],
                new_res=self.img_res)
            depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
            data_image = np.concatenate((data_image, depth), axis=-1).astype(float)
        #RGBD_image = (np.divide(RGBD_image, 255.) * 2.) - 1.
        #print(np.mean(RGBD_image))

        #data_image = data_image.swapaxes(1, 2).swapaxes(0, 1)

        #vis.plot_image(cropped_img)
        #vis.show()

        cropped_img_non_noisy = np.copy(cropped_img)
        cropped_img_noisy = util.add_noise(cropped_img, 0.2)
        data_image = util.add_noise(data_image, 0.25)
        if self.transform:
            data_image = self.transform(data_image)
            cropped_img_noisy = self.transform(cropped_img_noisy)
            cropped_img_non_noisy = self.transform(cropped_img_non_noisy)
            colour = self.transform(colour)

        return data_image, (colour, pose)

    def __len__(self):
        return self.length



def DataLoader(root_folder, transform=None, batch_size=4, img_res=(64, 64), num_channels=4):
    dataset = YCB_Dataset(root_folder, transform=transform, img_res=img_res, num_channels=num_channels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False)