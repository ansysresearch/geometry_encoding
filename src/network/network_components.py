import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, mid_channels, out_channels, act_fnc=nn.ReLU(),
                 bias=True, with_normalization=False, stride=1, padding=1, kernel_size=3):
        super().__init__()
        layers_list = [nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, stride=stride, bias=bias)]
        if with_normalization: layers_list.append(nn.BatchNorm2d(mid_channels))
        if act_fnc: layers_list.append(act_fnc)
        layers_list.append(nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))
        if with_normalization: layers_list.append(nn.BatchNorm2d(out_channels))
        if act_fnc: layers_list.append(act_fnc)
        self.double_conv = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, maxpool=True, **kwargs):
        super().__init__()
        layers_list = []
        if maxpool: layers_list.append(nn.MaxPool2d(2))
        layers_list.append(DoubleConv(in_channels, out_channels, out_channels, **kwargs))
        self.maxpool_conv = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels // 2, out_channels, **kwargs)

    def forward(self, x1, x2=None):
        x = self.up(x1)
        if x2 is not None:
            assert x1.shape[0] == 2 * x2.shape[0]
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)
