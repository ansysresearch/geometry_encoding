from src.network.network_components import *


class CNN0(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv1 = DoubleConv(n_channels, 128, 128)
        self.inconv2 = DoubleConv(128, 128, 128)
        self.outconv = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inconv1(x)
        x2 = self.inconv2(x1)
        xo = self.outconv(x2)
        return xo


class CNN1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inconv0 = DoubleConv(n_channels, 64, 64)
        self.inconv1 = DoubleConv(64, 64, 128)
        self.inconv2 = DoubleConv(128, 128, 128)
        self.inconv3 = DoubleConv(128, 128, 256)
        self.inconv4 = DoubleConv(256, 256, 256)

        self.inconv5 = DoubleConv(512, 256, 256)
        self.inconv6 = DoubleConv(384, 192, 128)
        self.inconv7 = DoubleConv(256, 128, 128)
        self.inconv8 = DoubleConv(192, 96, 64)

        self.outconv = OutConv(64, n_classes)

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


# U-Net structure with the original template @ https://github.com/milesial/Pytorch-UNet/tree/master/unet
class UNet(nn.Module):
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
        elif up_module == ConvTrans:
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
        super().__init__(n_channels_in, n_channels_out, down_module=Down, up_module=Up)


class UNet2(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down, up_module=ConvTrans)


class UNet3(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down2, up_module=Up)


class UNet4(UNet):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down2, up_module=ConvTrans)


class AutoEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, down_module=None, up_module=None):
        super().__init__()
        s1, s2, s3 = 64, 64, 64
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
        super().__init__(n_channels_in, n_channels_out, down_module=Down, up_module=Up)


class AutoEncoder2(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down, up_module=ConvTrans)


class AutoEncoder3(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down2, up_module=Up)


class AutoEncoder4(AutoEncoder):
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__(n_channels_in, n_channels_out, down_module=Down2, up_module=ConvTrans)


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

