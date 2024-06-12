import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


class Unet(nn.Module):
    start_channels = 64

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_sample_1 = DownSample(in_channels, self.start_channels)
        self.down_sample_2 = DownSample(self.start_channels, self.start_channels * 2)
        self.down_sample_3 = DownSample(self.start_channels * 2, self.start_channels * 4)
        self.down_sample_4 = DownSample(self.start_channels * 4, self.start_channels * 8)

        self.bottleneck = DoubleConv(self.start_channels * 8, self.start_channels * 16)

        self.up_sample_1 = UpSample(self.start_channels * 16, self.start_channels * 8)
        self.up_sample_2 = UpSample(self.start_channels * 8, self.start_channels * 4)
        self.up_sample_3 = UpSample(self.start_channels * 4, self.start_channels * 2)
        self.up_sample_4 = UpSample(self.start_channels * 2, self.start_channels)

        self.out = nn.Conv2d(self.start_channels, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x_down_1, p1 = self.down_sample_1(x)
        x_down_2, p2 = self.down_sample_2(p1)
        x_down_3, p3 = self.down_sample_3(p2)
        x_down_4, p4 = self.down_sample_4(p3)

        b = self.bottleneck(p4)

        x_up_1 = self.up_sample_1(b, x_down_4)
        x_up_2 = self.up_sample_2(x_up_1, x_down_3)
        x_up_3 = self.up_sample_3(x_up_2, x_down_2)
        x_up_4 = self.up_sample_4(x_up_3, x_down_1)

        x = self.out(x_up_4)

        return x