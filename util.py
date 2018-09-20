import numpy as np
import torch
import visualize as vis

def add_noise(image, blackout_prob):
    image_size = image.shape[0] * image.shape[1]
    keep_idxs = np.random.choice(2, image_size, p=[1 - blackout_prob, blackout_prob]).reshape((image.shape[0], image.shape[1])).astype(bool)
    image[keep_idxs, :] = 0

    return image

def add_noise_torch(image_batch, blackout_prob):
    image_size = image_batch.shape[2] * image_batch.shape[3]
    for i in range(image_batch.shape[0]):
        keep_idxs = np.random.choice(2, image_size, p=[1 - blackout_prob, blackout_prob]).\
            reshape((image_batch.shape[2], image_batch.shape[3])).astype(bool)
        image_numpy = image_batch[i].numpy()
        image_numpy[:, keep_idxs] = 0
        image_batch[i] = torch.from_numpy(image_numpy)
    return image_batch

# https://pytorch.org/docs/stable/nn.html#convolution-layers
def conv2d_output_size(h_in, w_in, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    h_out = int(( (h_in + (2*padding[0]) - (dilation[0]*kernel[0] - 1) - 1)/stride[0]) + 1)
    w_out = int(((w_in + (2 * padding[1]) - (dilation[1] * kernel[1] - 1) - 1) / stride[1]) + 1)
    return h_out, w_out

def conv2d_output_size_from_layer(h_in, w_in, conv2d_layer):
    kernel = conv2d_layer.kernel_size
    stride = conv2d_layer.stride
    padding = conv2d_layer.padding
    dilation = conv2d_layer.dilation
    return conv2d_output_size(h_in, w_in, kernel, stride, padding, dilation)