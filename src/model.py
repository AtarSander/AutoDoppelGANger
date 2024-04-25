import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g, device):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim, features_g*16, kernel_size=4, stride=1, padding=0),
            self._block(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
            self._block(features_g*8, features_g*4,kernel_size=4, stride=2, padding=1),
            self._block(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        if device.type == "cuda":
            self.cuda()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, device):
        super(Discriminator, self).__init__()
        self.dsc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
        if device.type == "cuda":
            self.cuda()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.dsc(x)

