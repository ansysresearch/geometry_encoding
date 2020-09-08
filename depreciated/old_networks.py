class SiLU(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, input):
        return input * torch.sigmoid(input)


class Sin(nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, input):
        return torch.sin(input)


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


class Flatten(nn.Module):
    def forward(self, x):
        N = x.size()[0]
        return x.view(N, -1)

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256+128, 64)
        self.up4 = Up(128+64, 64)
        self.up5 = Up(128, 32)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512+64, 256)
        self.up2 = Up(256+64, 128)
        self.up3 = Up(128+64, 64)
        self.up4 = Up(64+64, 64)
        self.outc1 = OutConv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x1)
        x7 = self.up2(x6, x1)
        x8 = self.up3(x7, x1)
        x9 = self.up4(x8, x1)
        x10 = self.outc1(x9)
        return x10


class UNet5(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels, n_classes = 1, 1
        channels = [64, 64, 64, 64, 64]
        self.inc = DoubleConv(n_channels, channels[0], channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.up11 = Up(channels[0] + channels[1], channels[1])
        self.up12 = Up(channels[1] + channels[2], channels[2])
        self.up13 = Up(channels[2] + channels[3], channels[3])
        self.up14 = Up(channels[3] + channels[4], channels[4])

        self.up21 = Up(channels[1] + channels[2], channels[2])
        self.up22 = Up(channels[2] + channels[3], channels[3])
        self.up23 = Up(channels[3] + channels[4], channels[4])

        self.up31 = Up(channels[2] + channels[3], channels[3])
        self.up32 = Up(channels[3] + channels[4], channels[4])

        self.up41 = Up(channels[3] + channels[4], channels[4])
        self.outc = OutConv(channels[4], n_classes)

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)

        x01 = self.up11(x10, x00)
        x11 = self.up12(x20, x10)
        x21 = self.up13(x30, x20)
        x31 = self.up14(x40, x30)

        x02 = self.up21(x11, x01)
        x12 = self.up22(x21, x11)
        x22 = self.up23(x31, x21)

        x03 = self.up31(x12, x02)
        x13 = self.up32(x22, x12)

        x04 = self.up41(x13, x03)
        out = self.outc(x04)
        return out


class UNet6(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels, n_classes = 1, 1
        channels = [64, 128, 256, 128, 64]
        self.inc = DoubleConv(n_channels, channels[0], channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.up11 = Up2(channels[0] + channels[1], channels[1])
        self.up12 = Up2(channels[1] + channels[2], channels[2])
        self.up13 = Up2(channels[2] + channels[3], channels[3])
        self.up14 = Up2(channels[3] + channels[4], channels[4])

        self.up21 = Up2(channels[0] + channels[1] + channels[2], channels[2])
        self.up22 = Up2(channels[1] + channels[2] + channels[3], channels[3])
        self.up23 = Up2(channels[2] + channels[3] + channels[4], channels[4])

        self.up31 = Up2(channels[0] + channels[1] + channels[2] + channels[3], channels[3])
        self.up32 = Up2(channels[1] + channels[2] + channels[3] + channels[4], channels[4])

        self.up41 = Up2(channels[0] + channels[1] + channels[2] + channels[3] + channels[4], channels[4])
        self.outc = OutConv(channels[4], n_classes)

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)

        x01 = self.up11([x10, x00])
        x11 = self.up12([x20, x10])
        x21 = self.up13([x30, x20])
        x31 = self.up14([x40, x30])

        x02 = self.up21([x11, x00, x01])
        x12 = self.up22([x21, x10, x11])
        x22 = self.up23([x31, x20, x21])

        x03 = self.up31([x12, x00, x01, x02])
        x13 = self.up32([x22, x10, x11, x12])

        x04 = self.up41([x13, x00, x01, x02, x03])
        out = self.outc(x04)
        return out


class UNet7(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels, n_classes = 1, 1
        channels = [64, 64, 128, 64, 64]
        self.inc = DoubleConv(n_channels, channels[0], channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])

        self.up11 = Up2(channels[0] + channels[1], channels[1])
        self.up12 = Up2(channels[1] + channels[2], channels[2])
        self.up13 = Up2(channels[2] + channels[3], channels[3])
        self.up14 = Up2(channels[3] + channels[4], channels[4])

        self.up21 = Up2(channels[0] + channels[1] + channels[2], channels[2])
        self.up22 = Up2(channels[1] + channels[2] + channels[3], channels[3])
        self.up23 = Up2(channels[2] + channels[3] + channels[4], channels[4])

        self.up31 = Up2(channels[0] + channels[1] + channels[2] + channels[3], channels[3])
        self.up32 = Up2(channels[1] + channels[2] + channels[3] + channels[4], channels[4])

        self.up41 = Up2(channels[0] + channels[1] + channels[2] + channels[3] + channels[4], channels[4])
        self.outc = OutConv(channels[0] + channels[1] + channels[2] + channels[3] + channels[4], n_classes)

    def forward(self, x):
        x00 = self.inc(x)
        x10 = self.down1(x00)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)

        x01 = self.up11([x10, x00])
        x11 = self.up12([x20, x10])
        x21 = self.up13([x30, x20])
        x31 = self.up14([x40, x30])

        x02 = self.up21([x11, x00, x01])
        x12 = self.up22([x21, x10, x11])
        x22 = self.up23([x31, x20, x21])

        x03 = self.up31([x12, x00, x01, x02])
        x13 = self.up32([x22, x10, x11, x12])

        x04 = self.up41([x13, x00, x01, x02, x03])

        x_tot = torch.cat([x00, x01, x02, x03, x04], dim=1)
        out = self.outc(x_tot)
        return out

class Net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc1 = DoubleConv(n_channels, 32, 64)
        self.inc2 = DoubleConv(64, 128, 256)
        # self.inc3 = DoubleConv(256, 512, 1024)
        # self.inc4 = DoubleConv(1024, 512, 256)
        self.inc5 = DoubleConv(256, 128, 64)
        self.inc6 = DoubleConv(64, 32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc1(x)
        x2 = self.inc2(x1)
        # x4 = self.inc3(x3)
        # x5 = self.inc4(x4)
        x5 = x2
        x6 = self.inc5(x5)
        x7 = self.inc6(x6)
        x8 = self.outc(x7)
        return x8

