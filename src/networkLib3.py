import torch
import torch.nn as nn
import torch.nn.functional as F

dtype1 = torch.DoubleTensor
dtype2 = torch.float64

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x.view(N,-1)

class UnFlatten30(nn.Module):
    def forward(self,x):
        N,CHW = x.size()
        C = 32
        H = 16
        W = 16
        return x.view(N,C,H,W)

class UnFlatten34(nn.Module):
    def forward(self,x):
        N,CHW = x.size()
        C = 8
        H = 8
        W = 8
        return x.view(N,C,H,W)

class UnFlatten35(nn.Module):
    def forward(self,x):
        N,CHW = x.size()
        C = 4
        H = 8
        W = 8
        return x.view(N,C,H,W)

class UnFlatten36(nn.Module):
    def forward(self,x):
        N,CHW = x.size()
        C = 2
        H = 8
        W = 8
        return x.view(N,C,H,W)

class UnFlatten37(nn.Module):
    def forward(self,x):
        N,CHW = x.size()
        C = 1
        H = 8
        W = 8
        return x.view(N,C,H,W)

class Swish(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x)


def getNetwork(networkID):

    if networkID == 4:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_Aa1 = nn.Conv2d( 1,32,3)
                self.conv_Aa2 = nn.Conv2d(32,32,3)
                self.conv_Ab1 = nn.Conv2d(32,32,3,padding=1)
                self.conv_Ab2 = nn.Conv2d(32,32,3,padding=1)
                self.conv_Ab3 = nn.Conv2d(32,32,3,padding=1)
                self.conv_Ac1 = nn.Conv2d(32,32,3)
                self.conv_Ac2 = nn.Conv2d(32,32,3)

                self.pool   = nn.MaxPool2d(2,2)

                self.flatten = Flatten()

                self.fc_A1   = nn.Linear(6*6*32,100)
                self.fc_A2   = nn.Linear(   100,100)
                self.fc_A3   = nn.Linear(   100,100)

                self.fc_C1   = nn.Linear(102,100)
                self.fc_C2   = nn.Linear(100,100)
                self.fc_C3   = nn.Linear(100,100)
                self.fc_C4   = nn.Linear(100,  1)

            def forward(self, x1, x2):                      # [ 1x64x64]
                x1 =          (F.relu(self.conv_Aa1(x1)))   # [32x62x62]
                x1 =          (F.relu(self.conv_Aa2(x1)))   # [32x60x60]
                x1 =          (F.relu(self.conv_Ab1(x1)))   # [32x60x60]
                x1 = self.pool(F.relu(self.conv_Ab2(x1)))   # [32x30x30]
                x1 =          (F.relu(self.conv_Ab3(x1)))   # [32x30x30]
                x1 = self.pool(F.relu(self.conv_Ac1(x1)))   # [32x14x14]
                x1 = self.pool(F.relu(self.conv_Ac2(x1)))   # [32x 6x 6]

                x1 = self.flatten(x1)

                x1 = F.relu(self.fc_A1(x1))
                x1 = F.relu(self.fc_A2(x1))
                x1 = F.relu(self.fc_A3(x1))

                x3 = torch.cat([x1,x2],1)

                x3 = F.relu(self.fc_C1(x3))
                x3 = F.relu(self.fc_C2(x3))
                x3 = F.relu(self.fc_C3(x3))
                x3 = self.fc_C4(x3)

                return x3


    elif networkID == 7:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_Aa1 = nn.Conv2d( 1,32,3)
                self.conv_Aa2 = nn.Conv2d(32,32,3)
                self.conv_Ab1 = nn.Conv2d(32,32,3,padding=1)
                self.conv_Ab2 = nn.Conv2d(32,32,3,padding=1)
                self.conv_Ac1 = nn.Conv2d(32,32,3)
                self.conv_Ac2 = nn.Conv2d(32,32,3)

                self.pool   = nn.MaxPool2d(2,2)

                self.flatten = Flatten()

                self.fc_A1   = nn.Linear(6*6*32,100)
                self.fc_A2   = nn.Linear(   100,100)

                self.fc_C1   = nn.Linear(102,100)
                self.fc_C2   = nn.Linear(100,100)
                self.fc_C3   = nn.Linear(100,  1)

            def forward(self, x1, x2):                      # [ 1x64x64]
                x1 =          (F.relu(self.conv_Aa1(x1)))   # [32x62x62]
                x1 =          (F.relu(self.conv_Aa2(x1)))   # [32x60x60]
                x1 =          (F.relu(self.conv_Ab1(x1)))   # [32x60x60]
                x1 = self.pool(F.relu(self.conv_Ab2(x1)))   # [32x30x30]
                x1 = self.pool(F.relu(self.conv_Ac1(x1)))   # [32x14x14]
                x1 = self.pool(F.relu(self.conv_Ac2(x1)))   # [32x 6x 6]

                x1 = self.flatten(x1)

                x1 = F.relu(self.fc_A1(x1))
                x1 = F.relu(self.fc_A2(x1))

                x3 = torch.cat([x1,x2],1)

                x3 = F.relu(self.fc_C1(x3))
                x3 = F.relu(self.fc_C2(x3))
                x3 = self.fc_C3(x3)

                return x3

    elif networkID == 9:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 4,3)
                self.conv_A_03 = nn.Conv2d( 4, 8,3)
                self.conv_A_04 = nn.Conv2d( 8, 8,3)
                self.conv_A_05 = nn.Conv2d( 8,16,3)
                self.conv_A_06 = nn.Conv2d(16,16,3)
                self.conv_A_07 = nn.Conv2d(16,32,3)
                self.conv_A_08 = nn.Conv2d(32,32,3)
                self.conv_A_09 = nn.Conv2d(32, 8,3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 4x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [ 8x58x58]
                x1 = F.relu(self.conv_A_04(x1))   # [ 8x56x56]
                x1 = F.relu(self.conv_A_05(x1))   # [16x54x54]
                x1 = F.relu(self.conv_A_06(x1))   # [16x52x52]
                x1 = F.relu(self.conv_A_07(x1))   # [32x50x50]
                x1 = F.relu(self.conv_A_08(x1))   # [32x48x48]
                x1 = F.relu(self.conv_A_09(x1))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = F.relu(self.fc_A_01(x1))     # [ 4000]
                x1 = F.relu(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = F.relu(self.fc_C_01(x3))     # [ 1000]
                x3 = F.relu(self.fc_C_02(x3))     # [  250]
                x3 = F.relu(self.fc_C_03(x3))     # [  100]
                x3 = self.fc_C_04(x3)             # [    1]

                return x3

    elif networkID == 10:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3,bias=False)
                self.conv_A_02 = nn.Conv2d( 4, 4,3,bias=False)
                self.conv_A_03 = nn.Conv2d( 4, 8,3,bias=False)
                self.conv_A_04 = nn.Conv2d( 8, 8,3,bias=False)
                self.conv_A_05 = nn.Conv2d( 8,16,3,bias=False)
                self.conv_A_06 = nn.Conv2d(16,16,3,bias=False)
                self.conv_A_07 = nn.Conv2d(16,32,3,bias=False)
                self.conv_A_08 = nn.Conv2d(32,32,3,bias=False)
                self.conv_A_09 = nn.Conv2d(32, 8,3,bias=False)

                self.bn_A_01 = nn.BatchNorm2d( 4)
                self.bn_A_02 = nn.BatchNorm2d( 4)
                self.bn_A_03 = nn.BatchNorm2d( 8)
                self.bn_A_04 = nn.BatchNorm2d( 8)
                self.bn_A_05 = nn.BatchNorm2d(16)
                self.bn_A_06 = nn.BatchNorm2d(16)
                self.bn_A_07 = nn.BatchNorm2d(32)
                self.bn_A_08 = nn.BatchNorm2d(32)
                self.bn_A_09 = nn.BatchNorm2d( 8)

                self.do_A_01 = nn.Dropout2d(p=0.3)
                self.do_A_02 = nn.Dropout2d(p=0.3)
                self.do_A_03 = nn.Dropout2d(p=0.3)
                self.do_A_04 = nn.Dropout2d(p=0.3)
                self.do_A_05 = nn.Dropout2d(p=0.3)
                self.do_A_06 = nn.Dropout2d(p=0.3)
                self.do_A_07 = nn.Dropout2d(p=0.3)
                self.do_A_08 = nn.Dropout2d(p=0.3)
                self.do_A_09 = nn.Dropout2d(p=0.3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.bn_A_01(self.conv_A_01(x1))))   # [ 4x62x62]
                x1 = self.do_A_02(F.relu(self.bn_A_02(self.conv_A_02(x1))))   # [ 4x60x60]
                x1 = self.do_A_03(F.relu(self.bn_A_03(self.conv_A_03(x1))))   # [ 8x58x58]
                x1 = self.do_A_04(F.relu(self.bn_A_04(self.conv_A_04(x1))))   # [ 8x56x56]
                x1 = self.do_A_05(F.relu(self.bn_A_05(self.conv_A_05(x1))))   # [16x54x54]
                x1 = self.do_A_06(F.relu(self.bn_A_06(self.conv_A_06(x1))))   # [16x52x52]
                x1 = self.do_A_07(F.relu(self.bn_A_07(self.conv_A_07(x1))))   # [32x50x50]
                x1 = self.do_A_08(F.relu(self.bn_A_08(self.conv_A_08(x1))))   # [32x48x48]
                x1 = self.do_A_09(F.relu(self.bn_A_09(self.conv_A_09(x1))))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = F.relu(self.fc_A_01(x1))     # [ 4000]
                x1 = F.relu(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = F.relu(self.fc_C_01(x3))     # [ 1000]
                x3 = F.relu(self.fc_C_02(x3))     # [  250]
                x3 = F.relu(self.fc_C_03(x3))     # [  100]
                x3 = self.fc_C_04(x3)             # [    1]

                return x3

    elif networkID == 11:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3,bias=False)
                self.conv_A_02 = nn.Conv2d( 4, 4,3,bias=False)
                self.conv_A_03 = nn.Conv2d( 4, 8,3,bias=False)
                self.conv_A_04 = nn.Conv2d( 8, 8,3,bias=False)
                self.conv_A_05 = nn.Conv2d( 8,16,3,bias=False)
                self.conv_A_06 = nn.Conv2d(16,16,3,bias=False)
                self.conv_A_07 = nn.Conv2d(16,32,3,bias=False)
                self.conv_A_08 = nn.Conv2d(32,32,3,bias=False)
                self.conv_A_09 = nn.Conv2d(32, 8,3,bias=False)

                # self.bn_A_01 = nn.BatchNorm2d( 4)
                # self.bn_A_02 = nn.BatchNorm2d( 4)
                # self.bn_A_03 = nn.BatchNorm2d( 8)
                # self.bn_A_04 = nn.BatchNorm2d( 8)
                # self.bn_A_05 = nn.BatchNorm2d(16)
                # self.bn_A_06 = nn.BatchNorm2d(16)
                # self.bn_A_07 = nn.BatchNorm2d(32)
                # self.bn_A_08 = nn.BatchNorm2d(32)
                # self.bn_A_09 = nn.BatchNorm2d( 8)

                self.do_A_01 = nn.Dropout2d(p=0.3)
                self.do_A_02 = nn.Dropout2d(p=0.3)
                self.do_A_03 = nn.Dropout2d(p=0.3)
                self.do_A_04 = nn.Dropout2d(p=0.3)
                self.do_A_05 = nn.Dropout2d(p=0.3)
                self.do_A_06 = nn.Dropout2d(p=0.3)
                self.do_A_07 = nn.Dropout2d(p=0.3)
                self.do_A_08 = nn.Dropout2d(p=0.3)
                self.do_A_09 = nn.Dropout2d(p=0.3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                # x1 = self.do_A_01(F.relu(self.bn_A_01(self.conv_A_01(x1))))   # [ 4x62x62]
                # x1 = self.do_A_02(F.relu(self.bn_A_02(self.conv_A_02(x1))))   # [ 4x60x60]
                # x1 = self.do_A_03(F.relu(self.bn_A_03(self.conv_A_03(x1))))   # [ 8x58x58]
                # x1 = self.do_A_04(F.relu(self.bn_A_04(self.conv_A_04(x1))))   # [ 8x56x56]
                # x1 = self.do_A_05(F.relu(self.bn_A_05(self.conv_A_05(x1))))   # [16x54x54]
                # x1 = self.do_A_06(F.relu(self.bn_A_06(self.conv_A_06(x1))))   # [16x52x52]
                # x1 = self.do_A_07(F.relu(self.bn_A_07(self.conv_A_07(x1))))   # [32x50x50]
                # x1 = self.do_A_08(F.relu(self.bn_A_08(self.conv_A_08(x1))))   # [32x48x48]
                # x1 = self.do_A_09(F.relu(self.bn_A_09(self.conv_A_09(x1))))   # [ 8x46x46]

                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [ 4x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [ 4x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [ 8x58x58]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [ 8x56x56]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x54x54]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x52x52]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [32x50x50]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [32x48x48]
                x1 = self.do_A_09(F.relu(self.conv_A_09(x1)))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = F.relu(self.fc_A_01(x1))     # [ 4000]
                x1 = F.relu(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = F.relu(self.fc_C_01(x3))     # [ 1000]
                x3 = F.relu(self.fc_C_02(x3))     # [  250]
                x3 = F.relu(self.fc_C_03(x3))     # [  100]
                x3 = self.fc_C_04(x3)             # [    1]

                return x3

    elif networkID == 12:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3,bias=False)
                self.conv_A_02 = nn.Conv2d( 4, 4,3,bias=False)
                self.conv_A_03 = nn.Conv2d( 4, 8,3,bias=False)
                self.conv_A_04 = nn.Conv2d( 8, 8,3,bias=False)
                self.conv_A_05 = nn.Conv2d( 8,16,3,bias=False)
                self.conv_A_06 = nn.Conv2d(16,16,3,bias=False)
                self.conv_A_07 = nn.Conv2d(16,32,3,bias=False)
                self.conv_A_08 = nn.Conv2d(32,32,3,bias=False)
                self.conv_A_09 = nn.Conv2d(32, 8,3,bias=False)

                self.bn_A_01 = nn.BatchNorm2d( 4)
                self.bn_A_02 = nn.BatchNorm2d( 4)
                self.bn_A_03 = nn.BatchNorm2d( 8)
                self.bn_A_04 = nn.BatchNorm2d( 8)
                self.bn_A_05 = nn.BatchNorm2d(16)
                self.bn_A_06 = nn.BatchNorm2d(16)
                self.bn_A_07 = nn.BatchNorm2d(32)
                self.bn_A_08 = nn.BatchNorm2d(32)
                self.bn_A_09 = nn.BatchNorm2d( 8)

                # self.do_A_01 = nn.Dropout2d(p=0.3)
                # self.do_A_02 = nn.Dropout2d(p=0.3)
                # self.do_A_03 = nn.Dropout2d(p=0.3)
                # self.do_A_04 = nn.Dropout2d(p=0.3)
                # self.do_A_05 = nn.Dropout2d(p=0.3)
                # self.do_A_06 = nn.Dropout2d(p=0.3)
                # self.do_A_07 = nn.Dropout2d(p=0.3)
                # self.do_A_08 = nn.Dropout2d(p=0.3)
                # self.do_A_09 = nn.Dropout2d(p=0.3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                # x1 = self.do_A_01(F.relu(self.bn_A_01(self.conv_A_01(x1))))   # [ 4x62x62]
                # x1 = self.do_A_02(F.relu(self.bn_A_02(self.conv_A_02(x1))))   # [ 4x60x60]
                # x1 = self.do_A_03(F.relu(self.bn_A_03(self.conv_A_03(x1))))   # [ 8x58x58]
                # x1 = self.do_A_04(F.relu(self.bn_A_04(self.conv_A_04(x1))))   # [ 8x56x56]
                # x1 = self.do_A_05(F.relu(self.bn_A_05(self.conv_A_05(x1))))   # [16x54x54]
                # x1 = self.do_A_06(F.relu(self.bn_A_06(self.conv_A_06(x1))))   # [16x52x52]
                # x1 = self.do_A_07(F.relu(self.bn_A_07(self.conv_A_07(x1))))   # [32x50x50]
                # x1 = self.do_A_08(F.relu(self.bn_A_08(self.conv_A_08(x1))))   # [32x48x48]
                # x1 = self.do_A_09(F.relu(self.bn_A_09(self.conv_A_09(x1))))   # [ 8x46x46]

                x1 = F.relu(self.bn_A_01(self.conv_A_01(x1)))   # [ 4x62x62]
                x1 = F.relu(self.bn_A_02(self.conv_A_02(x1)))   # [ 4x60x60]
                x1 = F.relu(self.bn_A_03(self.conv_A_03(x1)))   # [ 8x58x58]
                x1 = F.relu(self.bn_A_04(self.conv_A_04(x1)))   # [ 8x56x56]
                x1 = F.relu(self.bn_A_05(self.conv_A_05(x1)))   # [16x54x54]
                x1 = F.relu(self.bn_A_06(self.conv_A_06(x1)))   # [16x52x52]
                x1 = F.relu(self.bn_A_07(self.conv_A_07(x1)))   # [32x50x50]
                x1 = F.relu(self.bn_A_08(self.conv_A_08(x1)))   # [32x48x48]
                x1 = F.relu(self.bn_A_09(self.conv_A_09(x1)))   # [ 8x46x46]

                # na = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_01(self.conv_A_01(x1)))   # [ 4x62x62]
                # nb = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_02(self.conv_A_02(x1)))   # [ 4x60x60]
                # nc = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_03(self.conv_A_03(x1)))   # [ 8x58x58]
                # nd = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_04(self.conv_A_04(x1)))   # [ 8x56x56]
                # ne = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_05(self.conv_A_05(x1)))   # [16x54x54]
                # nf = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_06(self.conv_A_06(x1)))   # [16x52x52]
                # ng = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_07(self.conv_A_07(x1)))   # [32x50x50]
                # nh = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(self.bn_A_08(self.conv_A_08(x1)))   # [32x48x48]
                # # x1 = F.relu(self.conv_A_09(x1))   # [ 8x46x46]
                # # x1 = F.relu(self.bn_A_09(self.conv_A_09(x1)))   # [ 8x46x46]
                #
                # n0 = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = self.conv_A_09(x1)   # [ 8x46x46]
                # n1 = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = self.bn_A_09(x1)   # [ 8x46x46]
                # n2 = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # x1 = F.relu(x1)   # [ 8x46x46]
                # n3 = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # # print('{:5.3f}  {:5.3f}  {:5.3f}  {:5.3f}  {:5.3f}  {:5.3f}  {:5.3f}  {:5.3f} | {:5.3f}  {:5.3f}  {:5.3f}  {:5.3f}'.format(
                # #     na,nb,nc,nd,ne,nf,ng,nh,n0,n1,n2,n3))

                # x1 = F.relu(self.bn_A_01(self.conv_A_01(x1)))   # [ 4x62x62]
                # x1 = F.relu(self.bn_A_02(self.conv_A_02(x1)))   # [ 4x60x60]
                # x1 = F.relu(self.conv_A_03(x1))   # [ 8x58x58]
                # x1 = F.relu(self.conv_A_04(x1))   # [ 8x56x56]
                # x1 = F.relu(self.conv_A_05(x1))   # [16x54x54]
                # x1 = F.relu(self.conv_A_06(x1))   # [16x52x52]
                # x1 = F.relu(self.conv_A_07(x1))   # [32x50x50]
                # x1 = F.relu(self.conv_A_08(x1))   # [32x48x48]
                # x1 = F.relu(self.conv_A_09(x1))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = F.relu(self.fc_A_01(x1))     # [ 4000]
                x1 = F.relu(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = F.relu(self.fc_C_01(x3))     # [ 1000]
                x3 = F.relu(self.fc_C_02(x3))     # [  250]
                x3 = F.relu(self.fc_C_03(x3))     # [  100]
                x3 = self.fc_C_04(x3)             # [    1]

                return x3

    elif networkID == 13:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 8x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [32x29x29]
                x1 = F.relu(self.conv_A_04(x1))   # [32x27x27]
                x1 = F.relu(self.conv_A_05(x1))   # [32x13x13]
                x1 = F.relu(self.conv_A_06(x1))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = F.relu(self.fc_C_01(x3))     # [1000]
                x3 = F.relu(self.fc_C_02(x3))     # [ 250]
                x3 = F.relu(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 14:  # 13 with do
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=0.4)
                self.do_A_02 = nn.Dropout2d(p=0.4)
                self.do_A_03 = nn.Dropout2d(p=0.4)
                self.do_A_04 = nn.Dropout2d(p=0.4)
                self.do_A_05 = nn.Dropout2d(p=0.4)
                self.do_A_06 = nn.Dropout2d(p=0.4)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [ 4x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [ 8x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = F.relu(self.fc_C_01(x3))     # [1000]
                x3 = F.relu(self.fc_C_02(x3))     # [ 250]
                x3 = F.relu(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 15:  # 14 with more conv
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_0a = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_0b = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_0c = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=0.4)
                self.do_A_02 = nn.Dropout2d(p=0.4)
                self.do_A_03 = nn.Dropout2d(p=0.4)
                self.do_A_0a = nn.Dropout2d(p=0.4)
                self.do_A_0b = nn.Dropout2d(p=0.4)
                self.do_A_04 = nn.Dropout2d(p=0.4)
                self.do_A_05 = nn.Dropout2d(p=0.4)
                self.do_A_0c = nn.Dropout2d(p=0.4)
                self.do_A_06 = nn.Dropout2d(p=0.4)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [ 4x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [ 8x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_0a(F.relu(self.conv_A_0a(x1)))   # [32x29x29]
                x1 = self.do_A_0b(F.relu(self.conv_A_0b(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_0c(F.relu(self.conv_A_0c(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = F.relu(self.fc_C_01(x3))     # [1000]
                x3 = F.relu(self.fc_C_02(x3))     # [ 250]
                x3 = F.relu(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 16:  # 14 with more fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=0.4)
                self.do_A_02 = nn.Dropout2d(p=0.4)
                self.do_A_03 = nn.Dropout2d(p=0.4)
                self.do_A_04 = nn.Dropout2d(p=0.4)
                self.do_A_05 = nn.Dropout2d(p=0.4)
                self.do_A_06 = nn.Dropout2d(p=0.4)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 500)
                self.fc_C_0a   = nn.Linear( 500, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [ 4x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [ 8x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = F.relu(self.fc_C_01(x3))     # [1000]
                x3 = F.relu(self.fc_C_02(x3))     # [ 500]
                x3 = F.relu(self.fc_C_0a(x3))     # [ 250]
                x3 = F.relu(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 17:  # more filters at the beginning
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.6

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_02 = nn.Conv2d(64,48,3)
                self.conv_A_03 = nn.Conv2d(48,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                # self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_05 = nn.Conv2d(32,32,3)
                self.conv_A_06 = nn.Conv2d(32,16,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                # self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_01   = nn.Linear(16*23*23,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 500)
                self.fc_C_0a   = nn.Linear( 500, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [48x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                # x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                # x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x25x25]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x23x23]

                # x1 = self.flatten(x1)             # [3872]
                x1 = self.flatten(x1)             # [8464]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = F.relu(self.fc_C_01(x3))     # [1000]
                x3 = F.relu(self.fc_C_02(x3))     # [ 500]
                x3 = F.relu(self.fc_C_0a(x3))     # [ 250]
                x3 = F.relu(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 18:
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_02 = nn.Conv2d(64,48,3)
                self.conv_A_03 = nn.Conv2d(48,32,3)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3)
                self.conv_A_06 = nn.Conv2d(32,32,3)
                self.conv_A_07 = nn.Conv2d(32,32,4,stride=2)
                self.conv_A_08 = nn.Conv2d(32,32,3)
                self.conv_A_09 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)
                self.do_A_07 = nn.Dropout2d(p=p_do)
                self.do_A_08 = nn.Dropout2d(p=p_do)
                self.do_A_09 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*21*21,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 500)
                self.fc_C_03   = nn.Linear( 500, 250)
                self.fc_C_04   = nn.Linear( 250, 100)
                self.fc_C_05   = nn.Linear( 100,   1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [48x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x58x58]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x56x56]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x54x54]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x52x52]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [32x25x25]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [32x23x23]
                x1 = self.do_A_09(F.relu(self.conv_A_09(x1)))   # [32x21x21]

                x1 = self.flatten(x1)             # [14112]

                x1 = F.relu(self.fc_A_01(x1))     # [ 2000]
                x1 = F.relu(self.fc_A_02(x1))     # [ 1000]
                x1 = F.relu(self.fc_A_03(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = F.relu(self.fc_C_01(x3))     # [ 1000]
                x3 = F.relu(self.fc_C_02(x3))     # [  500]
                x3 = F.relu(self.fc_C_03(x3))     # [  250]
                x3 = F.relu(self.fc_C_04(x3))     # [  100]
                x3 = self.fc_C_05(x3)             # [    1]

                return x3

    elif networkID == 19:  # 9 with Swish
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 4,3)
                self.conv_A_03 = nn.Conv2d( 4, 8,3)
                self.conv_A_04 = nn.Conv2d( 8, 8,3)
                self.conv_A_05 = nn.Conv2d( 8,16,3)
                self.conv_A_06 = nn.Conv2d(16,16,3)
                self.conv_A_07 = nn.Conv2d(16,32,3)
                self.conv_A_08 = nn.Conv2d(32,32,3)
                self.conv_A_09 = nn.Conv2d(32, 8,3)

                self.flatten = Flatten()

                self.swish     = Swish()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 4x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [ 8x58x58]
                x1 = F.relu(self.conv_A_04(x1))   # [ 8x56x56]
                x1 = F.relu(self.conv_A_05(x1))   # [16x54x54]
                x1 = F.relu(self.conv_A_06(x1))   # [16x52x52]
                x1 = F.relu(self.conv_A_07(x1))   # [32x50x50]
                x1 = F.relu(self.conv_A_08(x1))   # [32x48x48]
                x1 = F.relu(self.conv_A_09(x1))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = self.swish(self.fc_A_01(x1))     # [ 4000]
                x1 = self.swish(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = self.swish(self.fc_C_01(x3))     # [ 1000]
                x3 = self.swish(self.fc_C_02(x3))     # [  250]
                x3 = self.swish(self.fc_C_03(x3))     # [  100]
                x3 =            self.fc_C_04(x3)      # [    1]

                return x3

    elif networkID == 20:  # 9 with tanh
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 4,3)
                self.conv_A_03 = nn.Conv2d( 4, 8,3)
                self.conv_A_04 = nn.Conv2d( 8, 8,3)
                self.conv_A_05 = nn.Conv2d( 8,16,3)
                self.conv_A_06 = nn.Conv2d(16,16,3)
                self.conv_A_07 = nn.Conv2d(16,32,3)
                self.conv_A_08 = nn.Conv2d(32,32,3)
                self.conv_A_09 = nn.Conv2d(32, 8,3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(46*46*8,1000)
                self.fc_A_02   = nn.Linear(   1000, 800)

                self.fc_C_01   = nn.Linear( 802,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 4x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [ 8x58x58]
                x1 = F.relu(self.conv_A_04(x1))   # [ 8x56x56]
                x1 = F.relu(self.conv_A_05(x1))   # [16x54x54]
                x1 = F.relu(self.conv_A_06(x1))   # [16x52x52]
                x1 = F.relu(self.conv_A_07(x1))   # [32x50x50]
                x1 = F.relu(self.conv_A_08(x1))   # [32x48x48]
                x1 = F.relu(self.conv_A_09(x1))   # [ 8x46x46]

                x1 = self.flatten(x1)             # [16928]

                x1 = torch.tanh(self.fc_A_01(x1))     # [ 4000]
                x1 = torch.tanh(self.fc_A_02(x1))     # [ 1000]

                x3 = torch.cat([x1,x2],1)         # [ 1002]

                x3 = torch.tanh(self.fc_C_01(x3))     # [ 1000]
                x3 = torch.tanh(self.fc_C_02(x3))     # [  250]
                x3 = torch.tanh(self.fc_C_03(x3))     # [  100]
                x3 = self.fc_C_04(x3)             # [    1]

                return x3

    elif networkID == 21:   # 13 with Swish
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.flatten = Flatten()

                self.swish     = Swish()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 8x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [32x29x29]
                x1 = F.relu(self.conv_A_04(x1))   # [32x27x27]
                x1 = F.relu(self.conv_A_05(x1))   # [32x13x13]
                x1 = F.relu(self.conv_A_06(x1))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = self.swish(self.fc_A_01(x1))     # [2000]
                x1 = self.swish(self.fc_A_02(x1))     # [1000]
                x1 = self.swish(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = self.swish(self.fc_C_01(x3))     # [1000]
                x3 = self.swish(self.fc_C_02(x3))     # [ 250]
                x3 = self.swish(self.fc_C_03(x3))     # [ 100]
                x3 =            self.fc_C_04(x3)      # [   1]

                return x3

    elif networkID == 22:  # 13 with tanh
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                self.conv_A_01 = nn.Conv2d( 1, 4,3)
                self.conv_A_02 = nn.Conv2d( 4, 8,3)
                self.conv_A_03 = nn.Conv2d( 8,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_C_01   = nn.Linear(1002,1000)
                self.fc_C_02   = nn.Linear(1000, 250)
                self.fc_C_03   = nn.Linear( 250, 100)
                self.fc_C_04   = nn.Linear( 100,   1)

            def forward(self, x1, x2):            # [ 1x64x64]
                x1 = F.relu(self.conv_A_01(x1))   # [ 4x62x62]
                x1 = F.relu(self.conv_A_02(x1))   # [ 8x60x60]
                x1 = F.relu(self.conv_A_03(x1))   # [32x29x29]
                x1 = F.relu(self.conv_A_04(x1))   # [32x27x27]
                x1 = F.relu(self.conv_A_05(x1))   # [32x13x13]
                x1 = F.relu(self.conv_A_06(x1))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = torch.tanh(self.fc_A_01(x1))     # [2000]
                x1 = torch.tanh(self.fc_A_02(x1))     # [1000]
                x1 = torch.tanh(self.fc_A_03(x1))     # [1000]

                x3 = torch.cat([x1,x2],1)         # [1002]

                x3 = torch.tanh(self.fc_C_01(x3))     # [1000]
                x3 = torch.tanh(self.fc_C_02(x3))     # [ 250]
                x3 = torch.tanh(self.fc_C_03(x3))     # [ 100]
                x3 = self.fc_C_04(x3)             # [   1]

                return x3

    elif networkID == 30:  # encoder decoder binary map
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_02 = nn.Conv2d(64,64,3)
                self.conv_A_03 = nn.Conv2d(64,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_B_01   = nn.Linear(1000,32*16*16)

                self.unflatten = UnFlatten30()
                self.upsample  = nn.Upsample(scale_factor=2,mode='bilinear')

                self.conv_B_01 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_02 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_03 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_05 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_06 = nn.Conv2d(32, 1,3,padding=1)

                self.do_B_01 = nn.Dropout2d(p=p_do)
                self.do_B_02 = nn.Dropout2d(p=p_do)
                self.do_B_03 = nn.Dropout2d(p=p_do)
                self.do_B_04 = nn.Dropout2d(p=p_do)
                self.do_B_05 = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                # n0 = torch.sum(torch.isnan(x1)).item() / x1.numel()

                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                # n1 = torch.sum(torch.isnan(x1)).item() / x1.numel()

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                # n2 = torch.sum(torch.isnan(x1)).item() / x1.numel()
                # nw = torch.sum(torch.isnan(self.fc_B_01.weight)).item() / self.fc_B_01.weight.numel()
                # nb = torch.sum(torch.isnan(self.fc_B_01.bias)).item()   / self.fc_B_01.bias.numel()

                x1 = F.relu(self.fc_B_01(x1))     # [8192]
                # x1 = self.fc_B_01(x1)     # [8192]

                # n2a= torch.sum(torch.isnan(x1)).item() / x1.numel()

                # x1 = F.relu(x1)     # [8192]

                # n2b= torch.sum(torch.isnan(x1)).item() / x1.numel()

                x1 = self.unflatten(x1)                         # [32x16x16]

                # n3 = torch.sum(torch.isnan(x1)).item() / x1.numel()

                x1 = self.do_B_01(F.relu(self.conv_B_01(x1)))   # [32x16x16]
                x1 = self.upsample(x1)                          # [32x32x32]
                x1 = self.do_B_02(F.relu(self.conv_B_02(x1)))   # [32x32x32]
                x1 = self.do_B_03(F.relu(self.conv_B_03(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_B_04(F.relu(self.conv_B_04(x1)))   # [32x64x64]
                x1 = self.do_B_05(F.relu(self.conv_B_05(x1)))   # [32x64x64]
                x1 =                     self.conv_B_06(x1)     # [ 1x64x64]

                # n4 = torch.sum(torch.isnan(x1)).item() / x1.numel()


                # print('0:{:5.3f}  1:{:5.3f}  2:{:5.3f}  w:{:5.3f}  b:{:5.3f}  2a:{:5.3f}  2b:{:5.3f}  3:{:5.3f}  4:{:5.3f}'.format(
                #     n0,n1,n2,nw,nb,n2a,n2b,n3,n4))

                return x1

    elif networkID == 31:  # 30 with more conv at encoder
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_0a = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3)
                self.conv_A_03 = nn.Conv2d(64,32,4,stride=2)
                self.conv_A_0b = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_0c = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_0a = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_0b = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_0c = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_B_01   = nn.Linear(1000,32*16*16)

                self.unflatten = UnFlatten30()
                self.upsample  = nn.Upsample(scale_factor=2,mode='bilinear')

                self.conv_B_01 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_02 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_03 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_05 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_06 = nn.Conv2d(32, 1,3,padding=1)

                self.do_B_01 = nn.Dropout2d(p=p_do)
                self.do_B_02 = nn.Dropout2d(p=p_do)
                self.do_B_03 = nn.Dropout2d(p=p_do)
                self.do_B_04 = nn.Dropout2d(p=p_do)
                self.do_B_05 = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_0a(F.relu(self.conv_A_0a(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_0b(F.relu(self.conv_A_0b(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_0c(F.relu(self.conv_A_0c(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x1 = F.relu(self.fc_B_01(x1))     # [8192]

                x1 = self.unflatten(x1)                         # [32x16x16]

                x1 = self.do_B_01(F.relu(self.conv_B_01(x1)))   # [32x16x16]
                x1 = self.upsample(x1)                          # [32x32x32]
                x1 = self.do_B_02(F.relu(self.conv_B_02(x1)))   # [32x32x32]
                x1 = self.do_B_03(F.relu(self.conv_B_03(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_B_04(F.relu(self.conv_B_04(x1)))   # [32x64x64]
                x1 = self.do_B_05(F.relu(self.conv_B_05(x1)))   # [32x64x64]
                x1 =                     self.conv_B_06(x1)     # [ 1x64x64]

                return x1

    elif networkID == 32:  # 30 with more conv at decoder
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_02 = nn.Conv2d(64,64,3)
                self.conv_A_03 = nn.Conv2d(64,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_B_01   = nn.Linear(1000,32*16*16)

                self.unflatten = UnFlatten30()
                self.upsample  = nn.Upsample(scale_factor=2,mode='bilinear')

                self.conv_B_01 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_02 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_03 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_05 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_08 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_09 = nn.Conv2d(32, 1,3,padding=1)

                self.do_B_01 = nn.Dropout2d(p=p_do)
                self.do_B_02 = nn.Dropout2d(p=p_do)
                self.do_B_03 = nn.Dropout2d(p=p_do)
                self.do_B_04 = nn.Dropout2d(p=p_do)
                self.do_B_05 = nn.Dropout2d(p=p_do)
                self.do_B_06 = nn.Dropout2d(p=p_do)
                self.do_B_07 = nn.Dropout2d(p=p_do)
                self.do_B_08 = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x1 = F.relu(self.fc_B_01(x1))     # [8192]

                x1 = self.unflatten(x1)                         # [32x16x16]

                x1 = self.do_B_01(F.relu(self.conv_B_01(x1)))   # [32x16x16]
                x1 = self.do_B_02(F.relu(self.conv_B_02(x1)))   # [32x16x16]
                x1 = self.upsample(x1)                          # [32x32x32]
                x1 = self.do_B_03(F.relu(self.conv_B_03(x1)))   # [32x32x32]
                x1 = self.do_B_04(F.relu(self.conv_B_04(x1)))   # [32x32x32]
                x1 = self.do_B_05(F.relu(self.conv_B_05(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_B_06(F.relu(self.conv_B_06(x1)))   # [32x64x64]
                x1 = self.do_B_07(F.relu(self.conv_B_07(x1)))   # [32x64x64]
                x1 = self.do_B_08(F.relu(self.conv_B_08(x1)))   # [32x64x64]
                x1 =                     self.conv_B_09(x1)     # [ 1x64x64]

                return x1

    elif networkID == 33:  # 30 with more fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3)
                self.conv_A_02 = nn.Conv2d(64,64,3)
                self.conv_A_03 = nn.Conv2d(64,32,4,stride=2)
                self.conv_A_04 = nn.Conv2d(32,32,3)
                self.conv_A_05 = nn.Conv2d(32,32,3,stride=2)
                self.conv_A_06 = nn.Conv2d(32,32,3)

                self.do_A_01 = nn.Dropout2d(p=p_do)
                self.do_A_02 = nn.Dropout2d(p=p_do)
                self.do_A_03 = nn.Dropout2d(p=p_do)
                self.do_A_04 = nn.Dropout2d(p=p_do)
                self.do_A_05 = nn.Dropout2d(p=p_do)
                self.do_A_06 = nn.Dropout2d(p=p_do)

                self.flatten = Flatten()

                self.fc_A_01   = nn.Linear(32*11*11,2000)
                self.fc_A_02   = nn.Linear(    2000,1000)
                self.fc_A_0a   = nn.Linear(    1000,1000)
                self.fc_A_0b   = nn.Linear(    1000,1000)
                self.fc_A_0c   = nn.Linear(    1000,1000)
                self.fc_A_03   = nn.Linear(    1000,1000)

                self.fc_B_01   = nn.Linear(1000,32*16*16)

                self.unflatten = UnFlatten30()
                self.upsample  = nn.Upsample(scale_factor=2,mode='bilinear')

                self.conv_B_01 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_02 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_03 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_05 = nn.Conv2d(32,32,3,padding=1)
                self.conv_B_06 = nn.Conv2d(32, 1,3,padding=1)

                self.do_B_01 = nn.Dropout2d(p=p_do)
                self.do_B_02 = nn.Dropout2d(p=p_do)
                self.do_B_03 = nn.Dropout2d(p=p_do)
                self.do_B_04 = nn.Dropout2d(p=p_do)
                self.do_B_05 = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x62x62]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x60x60]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x29x29]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x27x27]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [32x13x13]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [32x11x11]

                x1 = self.flatten(x1)             # [3872]

                x1 = F.relu(self.fc_A_01(x1))     # [2000]
                x1 = F.relu(self.fc_A_02(x1))     # [1000]
                x1 = F.relu(self.fc_A_0a(x1))     # [1000]
                x1 = F.relu(self.fc_A_0b(x1))     # [1000]
                x1 = F.relu(self.fc_A_0c(x1))     # [1000]
                x1 = F.relu(self.fc_A_03(x1))     # [1000]

                x1 = F.relu(self.fc_B_01(x1))     # [8192]

                x1 = self.unflatten(x1)                         # [32x16x16]

                x1 = self.do_B_01(F.relu(self.conv_B_01(x1)))   # [32x16x16]
                x1 = self.upsample(x1)                          # [32x32x32]
                x1 = self.do_B_02(F.relu(self.conv_B_02(x1)))   # [32x32x32]
                x1 = self.do_B_03(F.relu(self.conv_B_03(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_B_04(F.relu(self.conv_B_04(x1)))   # [32x64x64]
                x1 = self.do_B_05(F.relu(self.conv_B_05(x1)))   # [32x64x64]
                x1 =                     self.conv_B_06(x1)     # [ 1x64x64]

                return x1

    elif networkID == 34:  # enc dec
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                self.unflatten = UnFlatten34()

                self.fc_B_01   = nn.Linear(512,512)
                self.fc_B_02   = nn.Linear(512,512)
                self.fc_B_03   = nn.Linear(512,512)

                self.conv_C_01 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_C_02 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_01   = nn.Dropout2d(p=p_do)
                self.do_C_02   = nn.Dropout2d(p=p_do)
                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]

                x1 = F.relu(self.fc_B_01(x1))                   # [     512]
                x1 = F.relu(self.fc_B_02(x1))                   # [     512]
                x1 = F.relu(self.fc_B_03(x1))                   # [     512]

                x1 = self.unflatten(x1)                         # [ 8x 8x 8]

                x1 = self.do_C_01(F.relu(self.conv_C_01(x1)))   # [ 8x 8x 8]
                x1 = self.do_C_02(F.relu(self.conv_C_02(x1)))   # [ 8x 8x 8]
                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 35:  # 34 down to 4x8x8
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 4,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                self.unflatten = UnFlatten35()

                self.fc_B_01   = nn.Linear(256,256)
                self.fc_B_02   = nn.Linear(256,256)
                self.fc_B_03   = nn.Linear(256,256)

                self.conv_C_01 = nn.Conv2d( 4, 8,3,padding=1)
                self.conv_C_02 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_01   = nn.Dropout2d(p=p_do)
                self.do_C_02   = nn.Dropout2d(p=p_do)
                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 4x 8x 8]

                x1 = self.flatten(x1)                           # [     256]

                x1 = F.relu(self.fc_B_01(x1))                   # [     256]
                x1 = F.relu(self.fc_B_02(x1))                   # [     256]
                x1 = F.relu(self.fc_B_03(x1))                   # [     256]

                x1 = self.unflatten(x1)                         # [ 4x 8x 8]

                x1 = self.do_C_01(F.relu(self.conv_C_01(x1)))   # [ 8x 8x 8]
                x1 = self.do_C_02(F.relu(self.conv_C_02(x1)))   # [ 8x 8x 8]
                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 36:  # 34 down to 2x8x8
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 2,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                self.unflatten = UnFlatten36()

                self.fc_B_01   = nn.Linear(128,128)
                self.fc_B_02   = nn.Linear(128,128)
                self.fc_B_03   = nn.Linear(128,128)

                self.conv_C_01 = nn.Conv2d( 2, 8,3,padding=1)
                self.conv_C_02 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_01   = nn.Dropout2d(p=p_do)
                self.do_C_02   = nn.Dropout2d(p=p_do)
                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 2x 8x 8]

                x1 = self.flatten(x1)                           # [     128]

                x1 = F.relu(self.fc_B_01(x1))                   # [     128]
                x1 = F.relu(self.fc_B_02(x1))                   # [     128]
                x1 = F.relu(self.fc_B_03(x1))                   # [     128]

                x1 = self.unflatten(x1)                         # [ 2x 8x 8]

                x1 = self.do_C_01(F.relu(self.conv_C_01(x1)))   # [ 8x 8x 8]
                x1 = self.do_C_02(F.relu(self.conv_C_02(x1)))   # [ 8x 8x 8]
                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 37:  # 34 down to 1x8x8
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 1,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                self.unflatten = UnFlatten37()

                self.fc_B_01   = nn.Linear(64,64)
                self.fc_B_02   = nn.Linear(64,64)
                self.fc_B_03   = nn.Linear(64,64)

                self.conv_C_01 = nn.Conv2d( 1, 8,3,padding=1)
                self.conv_C_02 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_01   = nn.Dropout2d(p=p_do)
                self.do_C_02   = nn.Dropout2d(p=p_do)
                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 1x 8x 8]

                x1 = self.flatten(x1)                           # [      64]

                x1 = F.relu(self.fc_B_01(x1))                   # [      64]
                x1 = F.relu(self.fc_B_02(x1))                   # [      64]
                x1 = F.relu(self.fc_B_03(x1))                   # [      64]

                x1 = self.unflatten(x1)                         # [ 1x 8x 8]

                x1 = self.do_C_01(F.relu(self.conv_C_01(x1)))   # [ 8x 8x 8]
                x1 = self.do_C_02(F.relu(self.conv_C_02(x1)))   # [ 8x 8x 8]
                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 38:  # 34 w/o fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')

                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 39:  # 38 w/ flatten & unflatten
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                self.unflatten = UnFlatten34()

                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x1 = self.unflatten(x1)                         # [ 8x 8x 8]

                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 40:  # 38 w/ more conv
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_0a = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_0b = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_0c = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)
                self.conv_A_0d = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_0a   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_0b   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_0c   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)
                self.do_A_0d   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')

                self.conv_C_03 = nn.Conv2d( 8,16,3,padding=1)
                self.conv_C_04 = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_0a = nn.Conv2d(16,16,3,padding=1)
                self.conv_C_05 = nn.Conv2d(16,32,3,padding=1)
                self.conv_C_06 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_0b = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_07 = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_0c = nn.Conv2d(32,32,3,padding=1)
                self.conv_C_08 = nn.Conv2d(32, 1,3,padding=1)

                self.do_C_03   = nn.Dropout2d(p=p_do)
                self.do_C_04   = nn.Dropout2d(p=p_do)
                self.do_C_0a   = nn.Dropout2d(p=p_do)
                self.do_C_05   = nn.Dropout2d(p=p_do)
                self.do_C_06   = nn.Dropout2d(p=p_do)
                self.do_C_0b   = nn.Dropout2d(p=p_do)
                self.do_C_07   = nn.Dropout2d(p=p_do)
                self.do_C_0c   = nn.Dropout2d(p=p_do)

            def forward(self, x1):                              # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.do_A_0a(F.relu(self.conv_A_0a(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.do_A_0b(F.relu(self.conv_A_0b(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.do_A_0c(F.relu(self.conv_A_0c(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_0d(F.relu(self.conv_A_0d(x1)))   # [ 8x 8x 8]

                x1 = self.upsample(x1)                          # [ 8x16x16]
                x1 = self.do_C_03(F.relu(self.conv_C_03(x1)))   # [16x16x16]
                x1 = self.do_C_04(F.relu(self.conv_C_04(x1)))   # [16x16x16]
                x1 = self.do_C_0a(F.relu(self.conv_C_0a(x1)))   # [16x16x16]
                x1 = self.upsample(x1)                          # [16x32x32]
                x1 = self.do_C_05(F.relu(self.conv_C_05(x1)))   # [32x32x32]
                x1 = self.do_C_06(F.relu(self.conv_C_06(x1)))   # [32x32x32]
                x1 = self.do_C_0b(F.relu(self.conv_C_0b(x1)))   # [32x32x32]
                x1 = self.upsample(x1)                          # [32x64x64]
                x1 = self.do_C_07(F.relu(self.conv_C_07(x1)))   # [32x64x64]
                x1 = self.do_C_0c(F.relu(self.conv_C_0c(x1)))   # [32x64x64]
                x1 =                     self.conv_C_08(x1)     # [ 1x64x64]

                return x1

    elif networkID == 50:  # transfer learning: encoder from 38 (39)
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                # self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                # self.unflatten = UnFlatten34()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_02   = nn.Linear(       512,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     512]
                x3 = F.relu(self.fc_B_02(x3))                   # [     512]
                x3 = F.relu(self.fc_B_03(x3))                   # [     512]
                x3 = F.relu(self.fc_B_04(x3))                   # [     256]
                x3 = F.relu(self.fc_B_05(x3))                   # [     128]
                x3 = F.relu(self.fc_B_06(x3))                   # [      64]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 51:  # transfer learning: encoder from 38 (39) with do on fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_A_do = 0.0
                p_B_do = 0.4

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_A_do)
                self.do_A_02   = nn.Dropout2d(p=p_A_do)
                self.do_A_03   = nn.Dropout2d(p=p_A_do)
                self.do_A_04   = nn.Dropout2d(p=p_A_do)
                self.do_A_05   = nn.Dropout2d(p=p_A_do)
                self.do_A_06   = nn.Dropout2d(p=p_A_do)
                self.do_A_07   = nn.Dropout2d(p=p_A_do)
                self.do_A_08   = nn.Dropout2d(p=p_A_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_02   = nn.Linear(       512,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

                self.do_B_01   = nn.Dropout(p=p_B_do)
                self.do_B_02   = nn.Dropout(p=p_B_do)
                self.do_B_03   = nn.Dropout(p=p_B_do)
                self.do_B_04   = nn.Dropout(p=p_B_do)
                self.do_B_05   = nn.Dropout(p=p_B_do)
                self.do_B_06   = nn.Dropout(p=p_B_do)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = self.do_B_01(F.relu(self.fc_B_01(x3)))     # [     512]
                x3 = self.do_B_02(F.relu(self.fc_B_02(x3)))     # [     512]
                x3 = self.do_B_03(F.relu(self.fc_B_03(x3)))     # [     512]
                x3 = self.do_B_04(F.relu(self.fc_B_04(x3)))     # [     256]
                x3 = self.do_B_05(F.relu(self.fc_B_05(x3)))     # [     128]
                x3 = self.do_B_06(F.relu(self.fc_B_06(x3)))     # [      64]
                x3 =                     self.fc_B_07(x3)       # [       1]

                return x3

    elif networkID == 52:  # tl 50 more and smaller fc layers
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,256)
                self.fc_B_02   = nn.Linear(       256,256)
                self.fc_B_03   = nn.Linear(       256,256)
                self.fc_B_04   = nn.Linear(       256,256)
                self.fc_B_0a   = nn.Linear(       256,256)
                self.fc_B_0b   = nn.Linear(       256,256)
                self.fc_B_0c   = nn.Linear(       256,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_0d   = nn.Linear(       128,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     256]
                x3 = F.relu(self.fc_B_02(x3))                   # [     256]
                x3 = F.relu(self.fc_B_03(x3))                   # [     256]
                x3 = F.relu(self.fc_B_04(x3))                   # [     256]
                x3 = F.relu(self.fc_B_0a(x3))                   # [     256]
                x3 = F.relu(self.fc_B_0b(x3))                   # [     256]
                x3 = F.relu(self.fc_B_0c(x3))                   # [     256]
                x3 = F.relu(self.fc_B_05(x3))                   # [     128]
                x3 = F.relu(self.fc_B_0d(x3))                   # [     128]
                x3 = F.relu(self.fc_B_06(x3))                   # [      64]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 54:  # tl 52 more and smaller fc layers
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,256)
                self.fc_B_02   = nn.Linear(       256,256)
                self.fc_B_03   = nn.Linear(       256,192)
                self.fc_B_04   = nn.Linear(       192,192)
                self.fc_B_0a   = nn.Linear(       192,192)
                self.fc_B_0b   = nn.Linear(       192,192)
                self.fc_B_0c   = nn.Linear(       192,192)
                self.fc_B_0d   = nn.Linear(       192,192)
                self.fc_B_05   = nn.Linear(       192,128)
                self.fc_B_0e   = nn.Linear(       128,128)
                self.fc_B_0f   = nn.Linear(       128,128)
                self.fc_B_0g   = nn.Linear(       128,128)
                self.fc_B_0h   = nn.Linear(       128,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     256]
                x3 = F.relu(self.fc_B_02(x3))                   # [     256]
                x3 = F.relu(self.fc_B_03(x3))                   # [     192]
                x3 = F.relu(self.fc_B_04(x3))                   # [     192]
                x3 = F.relu(self.fc_B_0a(x3))                   # [     192]
                x3 = F.relu(self.fc_B_0b(x3))                   # [     192]
                x3 = F.relu(self.fc_B_0c(x3))                   # [     192]
                x3 = F.relu(self.fc_B_0d(x3))                   # [     192]
                x3 = F.relu(self.fc_B_05(x3))                   # [     128]
                x3 = F.relu(self.fc_B_0e(x3))                   # [     128]
                x3 = F.relu(self.fc_B_0f(x3))                   # [     128]
                x3 = F.relu(self.fc_B_0g(x3))                   # [     128]
                x3 = F.relu(self.fc_B_0h(x3))                   # [     128]
                x3 = F.relu(self.fc_B_06(x3))                   # [      64]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 60:  # lightweight 50
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                # self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                # self.unflatten = UnFlatten34()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     512]
                x3 = F.relu(self.fc_B_04(x3))                   # [     256]
                x3 = F.relu(self.fc_B_05(x3))                   # [     128]
                x3 = F.relu(self.fc_B_06(x3))                   # [      64]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 61:  # lightweight 50
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                # self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                # self.unflatten = UnFlatten34()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_07   = nn.Linear(       256,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     512]
                x3 = F.relu(self.fc_B_04(x3))                   # [     256]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 62:  # lightweight 50
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                # self.upsample  = nn.Upsample(scale_factor=2,mode='nearest')
                self.flatten   = Flatten()
                # self.unflatten = UnFlatten34()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.fc_B_01(x3))                   # [     512]
                x3 = F.relu(self.fc_B_03(x3))                   # [     512]
                x3 = F.relu(self.fc_B_04(x3))                   # [     256]
                x3 = F.relu(self.fc_B_05(x3))                   # [     128]
                x3 = F.relu(self.fc_B_06(x3))                   # [      64]
                x3 =        self.fc_B_07(x3)                    # [       1]

                return x3

    elif networkID == 64:  # 50 with swish for fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()
                self.swish     = Swish()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_02   = nn.Linear(       512,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = self.swish(self.fc_B_01(x3))               # [     512]
                x3 = self.swish(self.fc_B_02(x3))               # [     512]
                x3 = self.swish(self.fc_B_03(x3))               # [     512]
                x3 = self.swish(self.fc_B_04(x3))               # [     256]
                x3 = self.swish(self.fc_B_05(x3))               # [     128]
                x3 = self.swish(self.fc_B_06(x3))               # [      64]
                x3 =            self.fc_B_07(x3)                # [       1]

                return x3

    elif networkID == 65:  # 50 with tanh for fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_02   = nn.Linear(       512,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = torch.tanh(self.fc_B_01(x3))               # [     512]
                x3 = torch.tanh(self.fc_B_02(x3))               # [     512]
                x3 = torch.tanh(self.fc_B_03(x3))               # [     512]
                x3 = torch.tanh(self.fc_B_04(x3))               # [     256]
                x3 = torch.tanh(self.fc_B_05(x3))               # [     128]
                x3 = torch.tanh(self.fc_B_06(x3))               # [      64]
                x3 =            self.fc_B_07(x3)                # [       1]

                return x3

    elif networkID == 66:  # 50 with bn relu fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512,bias=False)
                self.fc_B_02   = nn.Linear(       512,512,bias=False)
                self.fc_B_03   = nn.Linear(       512,512,bias=False)
                self.fc_B_04   = nn.Linear(       512,256,bias=False)
                self.fc_B_05   = nn.Linear(       256,128,bias=False)
                self.fc_B_06   = nn.Linear(       128, 64,bias=False)
                self.fc_B_07   = nn.Linear(        64,  1)

                self.bn_B_01 = nn.BatchNorm1d(512)
                self.bn_B_02 = nn.BatchNorm1d(512)
                self.bn_B_03 = nn.BatchNorm1d(512)
                self.bn_B_04 = nn.BatchNorm1d(256)
                self.bn_B_05 = nn.BatchNorm1d(128)
                self.bn_B_06 = nn.BatchNorm1d( 64)


            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(self.bn_B_01(self.fc_B_01(x3)))     # [     512]
                x3 = F.relu(self.bn_B_02(self.fc_B_02(x3)))     # [     512]
                x3 = F.relu(self.bn_B_03(self.fc_B_03(x3)))     # [     512]
                x3 = F.relu(self.bn_B_04(self.fc_B_04(x3)))     # [     256]
                x3 = F.relu(self.bn_B_05(self.fc_B_05(x3)))     # [     128]
                x3 = F.relu(self.bn_B_06(self.fc_B_06(x3)))     # [      64]
                x3 =                     self.fc_B_07(x3)       # [       1]

                return x3

    elif networkID == 67:  # 50 with bn tanh fc
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512,bias=False)
                self.fc_B_02   = nn.Linear(       512,512,bias=False)
                self.fc_B_03   = nn.Linear(       512,512,bias=False)
                self.fc_B_04   = nn.Linear(       512,256,bias=False)
                self.fc_B_05   = nn.Linear(       256,128,bias=False)
                self.fc_B_06   = nn.Linear(       128, 64,bias=False)
                self.fc_B_07   = nn.Linear(        64,  1)

                self.bn_B_01 = nn.BatchNorm1d(512)
                self.bn_B_02 = nn.BatchNorm1d(512)
                self.bn_B_03 = nn.BatchNorm1d(512)
                self.bn_B_04 = nn.BatchNorm1d(256)
                self.bn_B_05 = nn.BatchNorm1d(128)
                self.bn_B_06 = nn.BatchNorm1d( 64)


            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = torch.tanh(self.bn_B_01(self.fc_B_01(x3))) # [     512]
                x3 = torch.tanh(self.bn_B_02(self.fc_B_02(x3))) # [     512]
                x3 = torch.tanh(self.bn_B_03(self.fc_B_03(x3))) # [     512]
                x3 = torch.tanh(self.bn_B_04(self.fc_B_04(x3))) # [     256]
                x3 = torch.tanh(self.bn_B_05(self.fc_B_05(x3))) # [     128]
                x3 = torch.tanh(self.bn_B_06(self.fc_B_06(x3))) # [      64]
                x3 =                         self.fc_B_07(x3)   # [       1]

                return x3

    elif networkID == 68:  # 50 tanh on last 2 FC activations
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                p_do = 0.0

                self.conv_A_01 = nn.Conv2d( 1,64,3,padding=1)
                self.conv_A_02 = nn.Conv2d(64,64,3,padding=1)
                self.conv_A_03 = nn.Conv2d(64,32,3,padding=1)
                self.conv_A_04 = nn.Conv2d(32,32,3,padding=1)
                self.conv_A_05 = nn.Conv2d(32,16,3,padding=1)
                self.conv_A_06 = nn.Conv2d(16,16,3,padding=1)
                self.conv_A_07 = nn.Conv2d(16, 8,3,padding=1)
                self.conv_A_08 = nn.Conv2d( 8, 8,3,padding=1)

                self.do_A_01   = nn.Dropout2d(p=p_do)
                self.do_A_02   = nn.Dropout2d(p=p_do)
                self.do_A_03   = nn.Dropout2d(p=p_do)
                self.do_A_04   = nn.Dropout2d(p=p_do)
                self.do_A_05   = nn.Dropout2d(p=p_do)
                self.do_A_06   = nn.Dropout2d(p=p_do)
                self.do_A_07   = nn.Dropout2d(p=p_do)
                self.do_A_08   = nn.Dropout2d(p=p_do)

                self.pool      = nn.MaxPool2d(2,2)
                self.flatten   = Flatten()

                self.fc_B_01   = nn.Linear(2+ 8* 8* 8,512)
                self.fc_B_02   = nn.Linear(       512,512)
                self.fc_B_03   = nn.Linear(       512,512)
                self.fc_B_04   = nn.Linear(       512,256)
                self.fc_B_05   = nn.Linear(       256,128)
                self.fc_B_06   = nn.Linear(       128, 64)
                self.fc_B_07   = nn.Linear(        64,  1)

            def forward(self, x1, x2):                          # [ 1x64x64]
                x1 = self.do_A_01(F.relu(self.conv_A_01(x1)))   # [64x64x64]
                x1 = self.do_A_02(F.relu(self.conv_A_02(x1)))   # [64x64x64]
                x1 = self.pool(x1)                              # [64x32x32]
                x1 = self.do_A_03(F.relu(self.conv_A_03(x1)))   # [32x32x32]
                x1 = self.do_A_04(F.relu(self.conv_A_04(x1)))   # [32x32x32]
                x1 = self.pool(x1)                              # [32x16x16]
                x1 = self.do_A_05(F.relu(self.conv_A_05(x1)))   # [16x16x16]
                x1 = self.do_A_06(F.relu(self.conv_A_06(x1)))   # [16x16x16]
                x1 = self.pool(x1)                              # [16x 8x 8]
                x1 = self.do_A_07(F.relu(self.conv_A_07(x1)))   # [ 8x 8x 8]
                x1 = self.do_A_08(F.relu(self.conv_A_08(x1)))   # [ 8x 8x 8]

                x1 = self.flatten(x1)                           # [     512]
                x3 = torch.cat([x1,x2],1)                       # [     514]

                x3 = F.relu(    self.fc_B_01(x3))               # [     512]
                x3 = F.relu(    self.fc_B_02(x3))               # [     512]
                x3 = F.relu(    self.fc_B_03(x3))               # [     512]
                x3 = F.relu(    self.fc_B_04(x3))               # [     256]
                x3 = torch.tanh(self.fc_B_05(x3))               # [     128]
                x3 = torch.tanh(self.fc_B_06(x3))               # [      64]
                x3 =            self.fc_B_07(x3)                # [       1]

                return x3

    else:
        print('ERROR: CANNOT FIND ARCHITECTURE "{:}"'.format(networkID),flush=True)
        assert(False)

    return Net()


# method to print np array memory size
def npSize(obj):
    s = obj.size * obj.itemsize
    u = 'B'
    if s>1024:
        s/= 1024
        u = 'kB'
    if s>1024:
        s/= 1024
        u = 'MB'
    if s>1024:
        s/= 1024
        u = 'GB'
    if s>1024:
        s/= 1024
        u = 'TB'

    print('{:5.1f} {}'.format(s,u))


# method to print number of parameters
def printParamNumber(net):
    nt = sum(p.numel() for p in net.parameters())
    nl = sum(p.numel() for p in net.parameters() if p.requires_grad)
    ut = ''
    ul = ''

    if int(round(nt))>1000:
        nt = nt/1000
        ut = 'k'
    if int(round(nt))>1000:
        nt = nt/1000
        ut = 'm'
    if int(round(nt))>1000:
        nt = nt/1000
        ut = 'b'

    if int(round(nl))>1000:
        nl = nl/1000
        ul = 'k'
    if int(round(nl))>1000:
        nl = nl/1000
        ul = 'm'
    if int(round(nl))>1000:
        nl = nl/1000
        ul = 'b'

    print('  total number of parameters     : {:6.1f}{:}'.format(0.1*round(10*nt),ut))
    print('  trainable number of parameters : {:6.1f}{:}'.format(0.1*round(10*nl),ul))


# method to initialize weights
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weightsNoConvBias(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
