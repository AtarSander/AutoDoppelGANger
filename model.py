import torch
import torch.nn as nn


class Generator(nn.module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim, features_g*8, kernel_size=4, stride=1, padding=0),
            self._block(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            self._block(features_g*4, features_g*2,kernel_size=4, stride=2, padding=1),
            self._block(features_g*2, features_g, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Discriminator(nn.module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.dsc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
