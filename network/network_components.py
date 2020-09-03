import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return input * torch.sigmoid(input)


class Sin(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return torch.sin(input)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, mid_channels, out_channels, act_fnc=nn.ReLU(inplace=True)):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            act_fnc,
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            #nn.BatchNorm2d(out_channels),
            act_fnc
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels // 2, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, in_channels // 2, out_channels)

    def forward(self, xvec):
        x0 = xvec[0]
        x1 = xvec[1]

        x0 = self.up(x0)

        # input is CHW
        diff_y = x1.size()[2] - x0.size()[2]
        diff_x = x1.size()[3] - x0.size()[3]
        if diff_y > 0 or diff_x > 0:
            x0 = F.pad(x0, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        xvec[0] = x0
        x = torch.cat(xvec, dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        N = x.size()[0]
        return x.view(N, -1)