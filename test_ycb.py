import torch
from auto_encoder2 import VQ_CVAE
import io_image
import numpy as np
import visualize as vis
import util
from torchvision import transforms
from torchvision.utils import save_image
import os

use_cuda = True
results_dir = './results/'
k = 256
hidden = 128
num_channels_in = 4
num_channels_out = 3
img_path = 'C:/Users/Administrator/Documents/Datasets/ycb_unreal_colour (493).png'
img_save_path = results_dir + 'output_img.png'
img_res = (640, 480)
noise_level = 0.0001
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5, 0.5))])

# load image
input_image = io_image.read_RGB_image(img_path, img_res)
if input_image.shape[2] == 4:
    input_image = input_image[:, :, 0:3]

data_image_noisy, noise_idxs = util.add_noise(input_image, noise_level)
data_image_noisy = transform(data_image_noisy).float()
noise_idxs = transform(noise_idxs).float()
data_image_noisy = torch.cat((data_image_noisy, noise_idxs), 0)

# load model
checkpoint = torch.load(results_dir + 'ycb_checkpoint.pth.tar')
args = checkpoint['args']
epoch = checkpoint['epoch']
model = VQ_CVAE(d=hidden, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
model.load_state_dict(checkpoint['state_dict'])
if use_cuda:
    print('Using Cuda')
    model.cuda()
    data_image_noisy = data_image_noisy.cuda()

data_image_noisy = data_image_noisy.view(1, num_channels_in, img_res[0], img_res[1])
outputs = model(data_image_noisy)

data_to_save = []
data_to_save.append(data_image_noisy[0, 0:3, :, :])
data_to_save = torch.stack(data_to_save).permute(0, 1, 3, 2).cpu()

outputs_to_save = []
outputs_to_save.append(outputs[0][0, 0:3, :, :])
outputs_to_save = torch.stack(outputs_to_save).permute(0, 1, 3, 2).cpu()

n = min(data_to_save.size(0), 1)
comparison = torch.cat([data_to_save[:n], outputs_to_save[:n]])

save_image(comparison, img_save_path, nrow=1, normalize=True)




