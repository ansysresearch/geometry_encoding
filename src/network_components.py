import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    r"""
    2 convolutional layers with/without Batch normalization

    Args:
        in_channels (int): number of input channels
        mid_channels (int): number of channels after first conv layer
        out_channels (int): number of output channels
        act_fnc (torch.nn function, optional): activation function, default :obj:`nn.ReLU()`
        bias (bool): use bias or not, default :obj:`True`)
        with_normalization (bool): use batch normalization or not, default :obj:`False`
        strided(bool): if :obj:`True`, first conv will have stride of 2 (e.g. for pooling), default :obj:`False`
    """

    def __init__(self,
                 in_channels, mid_channels,  out_channels, act_fnc=nn.ReLU(), bias=True,
                 with_normalization=False, strided=False):
        super().__init__()
        kernel_size, padding, stride = 3, 1, 1
        first_stride = 2 if strided else 1
        layers_list = [nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding, stride=first_stride, bias=bias)]
        if with_normalization: layers_list.append(nn.BatchNorm2d(mid_channels))
        if act_fnc: layers_list.append(act_fnc)
        layers_list.append(nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))
        if with_normalization: layers_list.append(nn.BatchNorm2d(out_channels))
        if act_fnc: layers_list.append(act_fnc)
        self.double_conv = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.double_conv(x)


class MaxDown(nn.Module):
    r"""Downscaling with maxpool then double conv layers

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        **kwargs (optional): additional arguments of :class:`DoubleConv`.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        layers_list = []
        layers_list.append(nn.MaxPool2d(2))
        layers_list.append(DoubleConv(in_channels, out_channels, out_channels, strided=False, **kwargs))
        self.pool = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.pool(x)


class ConvDown(nn.Module):
    r"""Downscaling with a stride double conv layers

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        **kwargs (optional): additional arguments of :class:`DoubleConv`.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.pool = DoubleConv(in_channels, out_channels, out_channels, strided=True, **kwargs)

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Upscaling with Upsampling and then double conv

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        **kwargs (optional): additional arguments of :class:`DoubleConv`.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, (in_channels + out_channels) // 2, out_channels, **kwargs)

    def forward(self, x1, x2=None):
        # x2 is the concatenated input for the UNet. Default is :obj:`None` for no concatenation
        x = self.up(x1)
        if x2 is not None:
            assert x.shape[-1] == x2.shape[-1] and x.shape[-2] == x2.shape[-2], \
                "concatenated inputs should have same shapes in the last two dimensions"
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class ConvUp(nn.Module):
    r"""Upscaling using transposed convolution and then a double conv layer
    
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            concat_size (None or int): number of concatenated input channels. If :obj:`None`, assume no concatenation
            **kwargs (optional): additional arguments of :class:`DoubleConv`
    """

    def __init__(self, in_channels, out_channels, concat_size=None, **kwargs):
        super().__init__()
        self.concat_size = concat_size
        if concat_size:
            self.convt = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False)
            self.conv = DoubleConv(concat_size, (concat_size + out_channels) // 2, out_channels, **kwargs)
        else:
            self.convt = nn.ConvTranspose2d(in_channels, in_channels,  kernel_size=2, stride=2, padding=0, bias=False)
            self.conv = DoubleConv(in_channels, (in_channels + out_channels) // 2, out_channels, **kwargs)

    def forward(self, x1, x2=None):
        assert (self.concat_size is None) is (x2 is None)
        x = self.convt(x1)
        if x2 is not None:
            assert x.shape[-1] == x2.shape[-1] and x.shape[-2] == x2.shape[-2], \
                "concatenated inputs should have same shapes in the last two dimensions"
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    r"""a conv layer with kernel size 1 to produce the output

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=True)

    def forward(self, x):
        return self.conv(x)
