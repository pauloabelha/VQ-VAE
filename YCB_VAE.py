import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util

class YCB_VAE(nn.Module):

    num_img_channels = 4
    latent_space_prop = .5
    conv1_kernel = 4
    conv2_kernel = 4
    indices_maxpool1 = []

    def __init__(self, batch_size, img_res):
        super(YCB_VAE, self).__init__()
        self.batch_size = batch_size
        self.img_res = img_res
        self.latent_space_size = int(self.img_res[0] * self.img_res[1] * self.latent_space_prop)
        # encoder
        self.conv1 = nn.Conv2d(self.num_img_channels, self.conv1_kernel, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_kernel)
        self.conv2 = nn.Conv2d(self.conv1_kernel, self.conv2_kernel, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_kernel)
        h_out, w_out = util.conv2d_output_size_from_layer(self.img_res[0], self.img_res[1], self.conv1)
        h_out, w_out = util.conv2d_output_size_from_layer(h_out, w_out, self.conv1)
        self.fc1 = nn.Linear(self.conv2_kernel * h_out * w_out, self.latent_space_size)
        self.fc2 = nn.Linear(self.latent_space_size, self.latent_space_size)

        # decoder
        self.de_fc1 = nn.Linear(self.latent_space_size, self.conv2_kernel * h_out * w_out)
        self.de_conv1 = nn.Conv2d(self.conv2_kernel, self.conv1_kernel, kernel_size=5, padding=2)
        self.de_batch_norm1 = nn.BatchNorm2d(self.conv2_kernel)
        self.de_conv2 = nn.Conv2d(self.conv1_kernel, self.num_img_channels, kernel_size=5, padding=2)
        self.de_batch_norm2 = nn.BatchNorm2d(self.num_img_channels)

    def encode(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = F.relu(out)
        h_out, w_out = util.conv2d_output_size_from_layer(x.shape[2], x.shape[3], self.conv2)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = out.view(-1, self.conv2.out_channels * h_out * w_out)
        out = self.fc1(out)
        mu = F.relu(out)
        logvar = self.fc2(out)
        logvar = F.relu(logvar)
        logvar = F.logsigmoid(logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        out = self.de_fc1(x)
        out = F.relu(out)
        out = self.de_conv1(out.view((-1, self.conv2.out_channels, self.img_res[0], self.img_res[0])))
        out = self.de_batch_norm1(out)
        out = F.relu(out)
        out = self.de_conv2(out)
        out = self.de_batch_norm2(out)
        out = F.relu(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar