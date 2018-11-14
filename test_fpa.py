import torch
from auto_encoder2 import VQ_CVAE
from torchvision import transforms
import fpa_dataset
import argparse
import visualize as vis
import io_image
import numpy as np
import fpa_io

use_cuda = True
results_dir = './results/'
k = 512
hidden = 256
num_channels_in = 1
num_channels_out = 1
img_path = 'C:/Users/Administrator/Documents/Datasets/ycb_unreal_colour (493).png'
img_save_path = results_dir + 'output_img.png'
img_res = (200, 200)

parser = argparse.ArgumentParser(description='Variational AutoEncoders')
parser.add_argument('--data-dir', default='/home/paulo/datasets/',
                                 help='directory containing the dataset')
parser.add_argument('--split-filename', default='', help='Dataset split filename')
args = parser.parse_args()

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5], std=[0.5])])

train_loader = fpa_dataset.DataLoaderReconstruction(root_folder=args.data_dir,
                                                      type='train', transform_color=None,
                                                      transform_depth=transforms,
                                                      batch_size=1,
                                                      split_filename=args.split_filename,
                                                      for_autoencoding=True,
                                                            input_type="depth",)
test_loader = fpa_dataset.DataLoaderReconstruction(root_folder=args.data_dir,
                                      type='test', transform_color=None,
                                      transform_depth=transforms,
                                      batch_size=1,
                                      split_filename=args.split_filename,
                                                     for_autoencoding=True,
                                                            input_type="depth")

# load model
results_rootpath = 'C:/Users/Administrator/Documents/GitHub/VQ-VAE/results/2018-11-02_19-04-06_cvpr2019/'
checkpoint = torch.load(results_rootpath + 'ycb_checkpoint.pth.tar')
args = checkpoint['args']
epoch = checkpoint['epoch']
model = VQ_CVAE(d=hidden, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
model.load_state_dict(checkpoint['state_dict'])
if use_cuda:
    print('Using Cuda')
    model.cuda()

for batch_idx, (data, label_img) in enumerate(train_loader):

    subpath, file_num = train_loader.dataset.get_subpath_and_file_num(batch_idx)

    if use_cuda:
        data = data.cuda()
        label_img = label_img.cuda()
    outputs = model(data)

    save_img_path = train_loader.dataset.root_folder +\
                    train_loader.dataset.gen_obj_folder +\
                    subpath + str(int(file_num)) + '_recon.npy'

    #vis.plot_image(data.cpu().numpy().reshape((200, 200)), title=subpath + '/' + str(file_num))
    #vis.show()

    output_img = outputs[0][0, 0:3, :, :].detach().cpu().numpy().reshape((200, 200))
    output_img *= train_loader.dataset.normalise_const_max_depth
    np.save(save_img_path, output_img)

    print('Train loader {} / {} ; {}'.format(batch_idx, len(train_loader), save_img_path))

    #output_img_loaded = np.load(save_img_path)

    #vis.plot_image(output_img_loaded)
    #vis.show()
    #vis.plot_image(output_img, title=subpath + '/' + str(file_num))
    #vis.show()

for batch_idx, (data, label_img) in enumerate(test_loader):

    subpath, file_num = train_loader.dataset.get_subpath_and_file_num(batch_idx)

    if use_cuda:
        data = data.cuda()
        label_img = label_img.cuda()
    outputs = model(data)

    save_img_path = train_loader.dataset.root_folder + \
                    train_loader.dataset.gen_obj_folder + \
                    subpath + str(int(file_num)) + '_recon.npy'

    output_img = outputs[0][0, 0:3, :, :].detach().cpu().numpy().reshape((200, 200))
    output_img *= train_loader.dataset.normalise_const_max_depth
    np.save(save_img_path, output_img)

    print('Test loader {} / {} ; {}'.format(batch_idx, len(train_loader), save_img_path))


