import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

import time

import numpy as np
import matplotlib.pyplot as plt

import argparse


print('')


# parse input parameters

exec(open("parseArgs.py").read())


# check cuda availability

if torch.cuda.is_available():
    device = torch.device(gpuName)
    print("Running on GPU {:1.0f}".format(args.gpu),flush=True)
else:
    device = torch.device("cpu")
    print("Running on the CPU",flush=True)


# load and initialize network

exec(open("networkLib3.py").read())
net = getNetwork(networkID).double().to(device)
try:
    torch.manual_seed(1234)
    net.apply(init_weights);
except:
    torch.manual_seed(1234)
    net.apply(init_weightsNoConvBias);


# loss and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=leRa, momentum=0.9)


# load data and prepare for training

X1  = np.load(dataPath + '{:02.0f}'.format(datasetID) + '_indi.npy')

exec(open("EdPrep.py").read())


# training the network

exec(open("EdTrain.py").read())
print('Finished Training',flush=True)


# save the network parameters

state = {
    'lossTrain': lossTrain,
    'lossTest': lossTest,
    'learnRate': learnRate,
    'epochID': epochID,
    'state_dict': net.state_dict(),
    'optimizer': optimizer.state_dict()
}
if resamplYes:
    state.update({
        'iUnShuffle':iUnShuffle,
        'tt_loss_tr_ep':tt_loss_tr_ep,
    })
torch.save(state,savePath+cName+'_params.pth')
