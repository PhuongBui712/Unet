import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = DoubleConv(in_channels, out_channels)
        self.pool_op = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv_x = self.conv_op(x)
        pool_x = self.pool_op(conv_x)

        return conv_x, pool_x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv_op = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_op = DoubleConv(in_channels, out_channels)

    def forward(self, x, skipped_x):
        x = self.up_conv_op(x)
        print(f"Before concat: x - {x.shape}, skipped x - {skipped_x.shape}")
        x = torch.cat([x, skipped_x], dim=1) # dim = 1 is channels dims of input (dim = 0 is batch size)
        print(f"After concat: x - {x.shape}, skipped x - {skipped_x.shape}")
        x = self.conv_op(x)

        return x