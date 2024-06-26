import torch
import torch.nn as nn
from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0):
        
        super().__init__()

        mid_channels = mid_channels or out_channels
        self.conv_ops = nn.Sequential(
            # first 
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=mid_channels),

            # second
            nn.Conv2d(in_channels=mid_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, X):
        return self.conv_ops(X)


class DownSample(nn.Module):
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        
    def forward(self, X):
        return self.pool(X)


class UpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0):
        
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
        
    def forward(self, X):
        return self.up_conv(X)
    

class CropAndConcat(nn.Module):
    def forward(self, X, contracting_X):
        contracting_X = transforms.functional.center_crop(
            img=contracting_X,
            output_size=(X.shape[2], X.shape[3])
        )
        X = torch.cat((X, contracting_X), dim=1)

        return X