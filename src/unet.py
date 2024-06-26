import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample, CropAndConcat


class Unet(nn.Module):
    # TODO: Customize the conv blocks for easy-scalable
    def __init__(self,
                 in_channels: int,
                 output_classes: int,
                 down_conv_kwargs: dict = None,
                 down_sample_kwargs: dict = None,
                 up_conv_kwargs: dict = None,
                 up_sample_kwargs: dict = None,
                 expansive_kwargs: dict = None):
        super().__init__()

        self.down_conv = nn.ModuleList([
            DoubleConv(in_channels=i, out_channels=o, **(down_conv_kwargs or {})) for i, o in ((in_channels, 64), (64, 128), (128, 256), (256, 512))
        ])

        self.down_sample = nn.ModuleList([
            DownSample(**(down_sample_kwargs or {})) for _ in range(4)
        ])

        self.up_conv = nn.ModuleList([
            DoubleConv(in_channels=i, out_channels=o, **(up_conv_kwargs or {})) for i, o in ((1024, 512), (512, 256), (256, 128), (128, 64))
        ])

        self.up_sample = nn.ModuleList([
            UpSample(in_channels=i, out_channels=o, **(up_sample_kwargs or {})) for i, o in ((1024, 512), (512, 256), (256, 128), (128, 64))
        ])

        self.crop_concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        self.bottlekneck = DoubleConv(in_channels=512,
                                      out_channels=1024,
                                      **(up_conv_kwargs or {}))
        
        self.output = nn.Conv2d(in_channels=64, out_channels=output_classes, kernel_size=1)
        
    def forward(self, X):
        pass_through = []
        for i in range(len(self.down_conv)):
            X = self.down_conv[i](X)
            pass_through = [X] + pass_through
            X = self.down_sample[i](X)

        X = self.bottlekneck(X)

        for i in range(len(self.up_conv)):
            X = self.up_sample[i](X)
            X = self.crop_concat[i](X, pass_through[i])
            print(X.shape)
            X = self.up_conv[i](X)

        X = self.output(X)

        return X