from src.network_components import *


class CNN1(nn.Module):
    r""" a convolutional layer with skip connection, but no pooling

    Args:
        n_channels_in (int): number of input channels
        n_channels_out (int): number of channels after first conv layer
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        s1, s2, s3 = 64, 128, 256
        self.inconv0 = DoubleConv(n_channels, s1, s1)
        self.inconv1 = DoubleConv(s1, s1, s2)
        self.inconv2 = DoubleConv(s2, s2, s2)
        self.inconv3 = DoubleConv(s2, s2, s3)
        self.inconv4 = DoubleConv(s3, s3, s3)

        self.inconv5 = DoubleConv(s3 + s3, s3, s3)
        self.inconv6 = DoubleConv(s3 + s2, s2 + s1, s2)
        self.inconv7 = DoubleConv(s2 + s2, s2, s2)
        self.inconv8 = DoubleConv(s2 + s1, s1, s1)

        self.outconv = OutConv(s1, n_classes)

    def forward(self, x):
        x1 = self.inconv0(x)
        x2 = self.inconv1(x1)
        x3 = self.inconv2(x2)
        x4 = self.inconv3(x3)
        x5 = self.inconv4(x4)
        x  = self.inconv5(torch.cat([x4, x5], dim=1))
        x  = self.inconv6(torch.cat([x3, x], dim=1))
        x  = self.inconv7(torch.cat([x2, x], dim=1))
        x  = self.inconv8(torch.cat([x1, x], dim=1))
        x  = self.outconv(x)
        return x


class UNet(nn.Module):
    r""" U-Net parent class
    original template @ https://github.com/milesial/Pytorch-UNet/tree/master/unet

    Args:
        n_channels_in (int): number of input channels
        n_channels_out (int): number of channels after first conv layer
        down_module (:class:`MaxDown` or :class:ConvDown`): the down-pooling layer
        up_module (:class:`Up` or :class:ConvUp`): the up-pooling layer
    """
    def __init__(self, n_channels_in, n_channels_out, down_module=None, up_module=None):
        super().__init__()
        s1, s2, s3 = 64, 128, 256
        self.inconv  = DoubleConv(n_channels_in, s1, s1)
        self.down1   = down_module(s1, s2)
        self.down2   = down_module(s2, s2)
        self.down3   = down_module(s2, s3)
        self.down4   = down_module(s3, s3)
        if up_module == Up:
            self.up1 = up_module(s3 + s3, s3)
            self.up2 = up_module(s3 + s2, s2)
            self.up3 = up_module(s2 + s2, s2)
            self.up4 = up_module(s2 + s1, s1)
        elif up_module == ConvUp:
            self.up1 = up_module(s3, s3, concat_size=s3 + s3)
            self.up2 = up_module(s3, s2, concat_size=s3 + s2)
            self.up3 = up_module(s2, s2, concat_size=s2 + s2)
            self.up4 = up_module(s2, s1, concat_size=s2 + s1)
        else:
            raise(ValueError("up_module %s is not recognized." % type(up_module)))

        self.outconv = OutConv(s1, n_channels_out)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        x  = self.outconv(x)
        return x


class UNet1(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=MaxDown, up_module=Up)


class UNet2(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=MaxDown, up_module=ConvUp)


class UNet3(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=ConvDown, up_module=Up)


class UNet4(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=ConvDown, up_module=ConvUp)


class AutoEncoder(nn.Module):
    r""" Auto Encoder model (similar to UNet, but with no skip connections)

    Args:
        n_channels_in (int): number of input channels
        n_channels_out (int): number of channels after first conv layer
        down_module (:class:`MaxDown` or :class:ConvDown`): the down-pooling layer
        up_module (:class:`Up` or :class:ConvUp`): the up-pooling layer
    """
    def __init__(self, n_channels_in, n_channels_out, down_module=None, up_module=None):
        super().__init__()
        s1, s2, s3 = 32, 32, 32
        self.inconv  = DoubleConv(n_channels_in, s1, s1)
        self.down1   = down_module(s1, s2)
        self.down2   = down_module(s2, s2)
        self.down3   = down_module(s2, s3)
        self.down4   = down_module(s3, s3)
        self.up1     = up_module(s3, s3)
        self.up2     = up_module(s3, s2)
        self.up3     = up_module(s2, s2)
        self.up4     = up_module(s2, s1)
        self.outconv = OutConv(s1, n_channels_out)

    def forward(self, x):
        x = self.inconv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.outconv(x)
        return x


class AutoEncoder1(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=MaxDown, up_module=Up)


class AutoEncoder2(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=MaxDown, up_module=ConvUp)


class AutoEncoder3(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=ConvDown, up_module=Up)


class AutoEncoder4(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=ConvDown, up_module=ConvUp)


class Interp2D(torch.nn.Module):
    r""" a 2D bilinear interpolation layer, with no learning parameters
        This layer interpolates from a regular grid data `f` of size n-by-n to a unstructured point list `xy`

        x0,y1,f01 ----------------------- x1, y1,f11
           |                                    |
           |                                    |
           |          x,y,?                     |
           |                                    |
           |                                    |
        x0, y0, f00 --------------------- x1, y0, f01


    Args:
        n (int): size of grid data
        xmin (float, optional) minimum value of x axis, default is `-1`
        xmax (float, optional) maximum value of x axis, default is `1`
        ymin (float, optional) minimum value of y axis, default is `-1`
        ymax (float, optional) maximum value of y axis, default is `1`

    Todo: variable sizing in x and y direction,
          remove for loop in finding f
    """
    def __init__(self, n, xmin=-1, xmax=1, ymin=-1, ymax=1):
        super().__init__()
        self.x = torch.linspace(xmin, xmax, n, requires_grad=True)
        self.y = torch.linspace(ymin, ymax, n, requires_grad=True)

    def check_inputs(self, f, xy):
        assert f.ndim == 3, "expect f to be a tensor of ndim 3"
        assert f.shape[1] == f.shape[2], "expect f to be a square tensor"
        assert xy.ndim == 3, "expect xy to be a tensor of ndim 3"
        assert xy.shape[0] == f.shape[0], "batch dimension should be indentical"

    def get_indices(self, xp, yp):
        idx = torch.searchsorted(self.x.detach(), xp.detach(), right=False)
        idy = torch.searchsorted(self.y.detach(), yp.detach(), right=False)
        return idx, idy

    def forward(self, f, xy):
        self.check_inputs(f, xy)
        xp = xy[..., 0].contiguous()
        yp = xy[..., 1].contiguous()
        ix, iy = self.get_indices(xp, yp)
        x0 = self.x[ix - 1]
        x1 = self.x[ix]
        y0 = self.y[iy - 1]
        y1 = self.y[iy]

        # Todo: do this witout any loop iteration
        bsize = f.shape[0]
        f00 = torch.stack([f[i, ix[i, ...] - 1, iy[i, ...] - 1] for i in range(bsize)])
        f01 = torch.stack([f[i, ix[i, ...] - 1, iy[i, ...]] for i in range(bsize)])
        f10 = torch.stack([f[i, ix[i, ...], iy[i, ...] - 1] for i in range(bsize)])
        f11 = torch.stack([f[i, ix[i, ...], iy[i, ...]] for i in range(bsize)])

        c1 = (xp - x0) / (x1 - x0)
        c2 = (yp - y0) / (y1 - y0)
        dx = f10 - f00
        dy = f01 - f00
        sol = f00 + c1 * dx + c2 * dy + c1 * c2 * (f11 - dx - dy - f00)
        return sol


def get_network(network_id):
    n_channels_in, n_channels_out = 1, 1
    if network_id == "UNet1":
        return UNet1(n_channels_in, n_channels_out)
    elif network_id == "UNet2":
        return UNet2(n_channels_in, n_channels_out)
    elif network_id == "UNet3":
        return UNet3(n_channels_in, n_channels_out)
    elif network_id == "UNet4":
        return UNet4(n_channels_in, n_channels_out)
    elif network_id == "CNN1":
        return CNN1(n_channels_in, n_channels_out)
    elif network_id == "AE1":
        return AutoEncoder1(n_channels_in, n_channels_out)
    elif network_id == "AE2":
        return AutoEncoder2(n_channels_in, n_channels_out)
    elif network_id == "AE3":
        return AutoEncoder3(n_channels_in, n_channels_out)
    elif network_id == "AE4":
        return AutoEncoder4(n_channels_in, n_channels_out)
    else:
        raise(IOError("Network ID is not recognized."))

