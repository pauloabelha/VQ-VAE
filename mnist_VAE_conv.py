import torch
import torch.nn as nn
import torch.nn.functional as F


class mnist_VAE_conv(nn.Module):
    def __init__(self):
        super(mnist_VAE_conv, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(8000, 50)
        self.fc21 = nn.Linear(50, 20)
        self.fc22 = nn.Linear(50, 20)

        # decoder
        self.de_fc2 = nn.Linear(20, 50)
        self.de_fc1 = nn.Linear(50, 8000)
        self.de_conv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.de_conv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)

    def encode(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(self.fc1(out.view(-1, 8000)))
        return self.fc21(out), self.fc22(out)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        out = self.de_fc2(x)
        out = F.relu(self.de_fc1(out))
        out = self.de_conv2(out.view((-1, 20, 20, 20)))
        out = self.de_conv1(out)
        out = F.sigmoid(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar