import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
from src.networkLib3 import *

# parse input parameters

networkID     = 68
caseName      = "jans_run"
datasetID     = 1
startNew      = True
oldEpoch      = 0
gpuName       = 'cuda:0'
nEpochs       = 100
frac_rsmpl    = 0
resamplYes    = False
saveEvery     = 20

batchSize     = 50
frac_train    = 0.8
leRa          = 1e-3
miniBatchSize = 50
printEvery    = 2

WL1Width      = 1.
useWL1        = True
eikonalFactor = 0.
useEik        = False


deltaNum = 1.0e-10  # finite difference stencil width for gradient estimation
torch.pi = torch.acos(torch.zeros(1)).item() * 2

assert((batchSize%miniBatchSize)==0)

savePath = '../data/network-parameters/'
dataPath = '../data/jans-data/'



# check cuda availability

if torch.cuda.is_available():
    device = torch.device(gpuName)
    print("Running on GPU %s" %gpuName, flush=True)
else:
    device = torch.device("cpu")
    print("Running on the CPU", flush=True)


# load and initialize network

net = getNetwork(networkID).double().to(device)
try:
    torch.manual_seed(1234)
    net.apply(init_weights)
except:
    torch.manual_seed(1234)
    net.apply(init_weightsNoConvBias)


# loss and optimizer

criterion = nn.L1Loss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=leRa, momentum=0.9)


# load data and prepare for training

X1  = np.load(dataPath + '{:02.0f}'.format(datasetID) + '_indi.npy'  )
SDF = np.load(dataPath + '{:02.0f}'.format(datasetID) + '_sdfc.npy'  )

X1[X1==0.0] = -1.0

print('')
print('X1  :  ({:4.0f},{:3.0f},{:3.0f})    '.format(
    X1.shape[0] ,X1.shape[1] ,X1.shape[2]),end='')
print('in [{:7.4f},{:7.4f},{:7.4f}]'.format(
    np.min(X1) ,np.mean(X1) ,np.max(X1)))
print('SDF :  ({:4.0f},{:3.0f},{:3.0f},{:1.0f})  '.format(
    SDF.shape[0],SDF.shape[1],SDF.shape[2],SDF.shape[3]),end='')
print('in [{:7.4f},{:7.4f},{:7.4f}]'.format(
    np.min(SDF),np.mean(SDF),np.max(SDF)))
print('')


print('neural network parameters')
printParamNumber(net)
print('')


frac_train_eff = frac_train*(1+frac_rsmpl) / (frac_train*(1+frac_rsmpl)+(1-frac_train))

n_geo      = int(X1.shape[0])
n_res_x    = int(X1.shape[1])
n_res_y    = int(X1.shape[2])
n_ep_dat   = int(SDF.shape[1])
n_samp     = int(SDF.shape[2])

n_geo_tr = int(frac_train*X1.shape[0])
n_geo_te = n_geo - n_geo_tr
n_geo_samp_tr = n_geo_tr*n_samp
n_geo_samp_te = n_geo_te*n_samp
n_res_geo_samp_tr = int(n_geo_samp_tr*frac_rsmpl)
n_tot_geo_samp_tr = n_geo_samp_tr + n_res_geo_samp_tr

nBperEtr   = int(n_tot_geo_samp_tr/batchSize)
nBperEte   = int(n_geo_samp_te    /batchSize)
nMBperB    = int(batchSize/miniBatchSize)
n_print_tr = int(nBperEtr/printEvery)
n_print_te = int(nBperEte/printEvery)


print('data set parameters')
print('  # of geometries              : {:5.0f}'.format(n_geo))
print('  # of samples per geo         : {:5.0f}'.format(n_samp))
print('  # of epochs with new samples : {:5.0f}'.format(n_ep_dat))
print('  image resolution             : {:2.0f}x{:2.0f}'.format(n_res_x,n_res_y))
print('')

print('training parameters')
print('  training fraction : {:5.3f}'.format(frac_train))
print('    # of training geo\'s : {:5.0f}'.format(n_geo_tr))
print('    # of testing geo\'s  : {:5.0f}'.format(n_geo_te))
print('    # of new training samples per epoch : {:6.0f}'.format(n_geo_samp_tr))
print('    # of new testing samples per epoch  : {:6.0f}'.format(n_geo_samp_te))
print('  resample fraction : {:5.3f}'.format(frac_rsmpl))
print('    effective training fraction               : {:6.3f}'.format(frac_train_eff))
print('    # of resampled training samples per epoch : {:6.0f}'.format(n_res_geo_samp_tr))
print('    total # of training samples per epoch     : {:6.0f}'.format(n_tot_geo_samp_tr))
print('  (mini) batch size : {:3.0f}  ({:3.0f})'.format(batchSize,miniBatchSize))
print('    # of training batches per epoch : {:4.0f}'.format(nBperEtr))
print('    # of testing batches per epoch  : {:4.0f}'.format(nBperEte))
print('    # of mini batches per batch     : {:4.0f}'.format(nMBperB))
print('  print every {:} batches'.format(printEvery))
print('    # of training prints : {:3.0f}'.format(n_print_tr))
print('    # of testing prints  : {:3.0f}'.format(n_print_te),flush=True)
print('')

print('loss parameters')
print('  use weighted L1     : {:}'.format(useWL1))
print('    weighted L1 width : {:6.3f}'.format(WL1Width))
print('  use eikonal loss : {:}'.format(useEik))
print('    eikonal factor : {:8.6f}'.format(eikonalFactor))


tt_x1_tr_ep   = torch.from_numpy(np.zeros((n_tot_geo_samp_tr,1,n_res_x,n_res_y)))
tt_x2_tr_ep   = torch.from_numpy(np.zeros((n_tot_geo_samp_tr,2)))
tt_y_tr_ep    = torch.from_numpy(np.zeros((n_tot_geo_samp_tr,1)))

tt_loss_tr_ep = torch.from_numpy(np.zeros((n_tot_geo_samp_tr,1)))

tt_x1_te_ep   = torch.from_numpy(np.zeros((n_geo_samp_te,1,n_res_x,n_res_y)))
tt_x2_te_ep   = torch.from_numpy(np.zeros((n_geo_samp_te,2)))
tt_y_te_ep    = torch.from_numpy(np.zeros((n_geo_samp_te,1)))


x1_dev = torch.from_numpy(np.zeros((batchSize,1,n_res_x,n_res_y))).to(device)
x2_dev = torch.from_numpy(np.zeros((batchSize,2))).to(device)
y_dev  = torch.from_numpy(np.zeros((batchSize,1))).to(device)

tt_X1  = torch.from_numpy(X1)
tt_SDF = torch.from_numpy(SDF)


resolutionID = n_res_x

cName = 'img{:03.0f}_arc{:02.0f}_dat{:02.0f}_{:}'.format(resolutionID,networkID,datasetID,caseName)
print('\n\nThis case\'s name is :  ' + cName,flush=True)

if startNew:
    print('Starting new')
    lossWL1Train = []
    lossEikTrain = []
    lossTrain    = []
    lossWL1Test  = []
    lossEikTest  = []
    lossTest     = []
    learnRate    = []
    epochID      = []
    np.random.seed(1234)
else:
    seName = cName + '_epo{:03.0f}'.format(oldEpoch)
    print('Starting from epoch {:}'.format(oldEpoch))
    state = torch.load(savePath+seName+'_params.pth')
    lossWL1Train  = state['lossWL1Train']
    lossEikTrain  = state['lossEikTrain']
    lossTrain     = state['lossTrain']
    lossWL1Test   = state['lossWL1Test']
    lossEikTest   = state['lossEikTest']
    lossTest      = state['lossTest']
    learnRate     = state['learnRate']
    epochID       = state['epochID']
    if resamplYes:
        iUnShuffle    = state['iUnShuffle']
        tt_loss_tr_ep = state['tt_loss_tr_ep'].detach()
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    np.random.seed(oldEpoch)
print('')


def printHeader(n_print_tr,n_print_te):
    print('    \t[    |',end='')
    for i in range(n_print_tr):
        print(' ',end='')
    print('|      |',end='')
    for i in range(n_print_te):
        print(' ',end='')
    print('|     ]')



# training the network

for param_group in optimizer.param_groups:
    param_group['lr'] = leRa
epochCount = len(epochID)
#printHeader(n_print_tr,n_print_te)
t_00 = time.time()


for epoch in range(nEpochs):
    i_ep_dat = (epochCount%n_ep_dat)
    print('{:04.0f}\t'.format(epochCount+1),end='',flush=True)


    t_0 = time.time()

    if resamplYes and epoch > 0:
        i0 = torch.argsort(tt_loss_tr_ep[iUnShuffle[:n_geo_samp_tr]],dim=0,descending=True)
        i1 = iUnShuffle[i0[:n_res_geo_samp_tr]][:,0]
        tt_x1_tr_ep[n_geo_samp_tr:] = tt_x1_tr_ep[i1]
        tt_x2_tr_ep[n_geo_samp_tr:] = tt_x2_tr_ep[i1]
        tt_y_tr_ep[n_geo_samp_tr:]  = tt_y_tr_ep[i1]

    for i in range(n_geo_tr):
        tt_x1_tr_ep[(i*n_samp):((i+1)*n_samp),0,:,:] = tt_X1[i,:,:]
        tt_x2_tr_ep[(i*n_samp):((i+1)*n_samp),:]     = tt_SDF[i,i_ep_dat,:,0:2]
        tt_y_tr_ep[(i*n_samp):((i+1)*n_samp),:]      = tt_SDF[i,i_ep_dat,:,2].reshape((n_samp,1))
    for i in range(n_geo_te):
        tt_x1_te_ep[(i*n_samp):((i+1)*n_samp),0,:,:] = tt_X1[i+n_geo_tr,:,:]
        tt_x2_te_ep[(i*n_samp):((i+1)*n_samp),:]     = tt_SDF[i+n_geo_tr,i_ep_dat,:,0:2]
        tt_y_te_ep[(i*n_samp):((i+1)*n_samp),:]      = tt_SDF[i+n_geo_tr,i_ep_dat,:,2].reshape((n_samp,1))

    if resamplYes and epoch == 0:
        i1 = np.random.permutation(n_geo_samp_tr)[:n_res_geo_samp_tr]
        tt_x1_tr_ep[n_geo_samp_tr:] = tt_x1_tr_ep[i1]
        tt_x2_tr_ep[n_geo_samp_tr:] = tt_x2_tr_ep[i1]
        tt_y_tr_ep[n_geo_samp_tr:]  = tt_y_tr_ep[i1]

    iShuffle   = np.random.permutation(n_tot_geo_samp_tr)
    iUnShuffle = np.argsort(iShuffle)
    tt_x1_tr_ep = tt_x1_tr_ep[iShuffle]
    tt_x2_tr_ep = tt_x2_tr_ep[iShuffle]
    tt_y_tr_ep  = tt_y_tr_ep[iShuffle]


    t_1 = time.time()

    #print('[{:04.1f}|'.format(t_1-t_0),end='',flush=True)
    net.train()
    lossWL1TrainE = 0.0
    lossEikTrainE = 0.0
    lossTrainE    = 0.0
    for iB in range(nBperEtr):
        optimizer.zero_grad()

        for iMB in range(nMBperB):
            iMB_ = range(iB*batchSize+iMB*miniBatchSize,iB*batchSize+(iMB+1)*miniBatchSize)

            x1_dev = tt_x1_tr_ep[iMB_].to(device)
            x2_dev = tt_x2_tr_ep[iMB_].to(device)
            y_dev  = tt_y_tr_ep[iMB_].to(device)
            p_dev = net(x1_dev,x2_dev)

            weights      = torch.ones_like(y_dev)
            weights_sum0 = weights.sum()
            if useWL1:
                # === uncomment 1 shape ===
                # (1) exponential decay shape
                # weights      = torch.exp(-torch.abs(y_dev)/WL1Width)
                # (2) cosine with exponential decay finish shape
                wcos = torch.cos(torch.abs(y_dev)/WL1Width*2.0/torch.pi)
                wcos[torch.abs(y_dev)>2.0*WL1Width*torch.pi] = -1.0
                wexp = torch.exp(-torch.abs(y_dev)/WL1Width)
                wsgm = torch.sigmoid((torch.abs(y_dev)-2.0*WL1Width)*3.0/WL1Width)
                weights = (1.0-wsgm)*wcos + wsgm*wexp
            weights_sum1 = weights.sum()
            weights      = weights*weights_sum0/weights_sum1

            lossWL1 = criterion(p_dev,y_dev)
            lossWL1 = lossWL1*weights
            lossWL1 = lossWL1.sum() / weights.sum()

            if useEik:
                x2_A = x2_dev.clone()
                x2_B = x2_dev.clone()
                x2_C = x2_dev.clone()
                x2_D = x2_dev.clone()
                x2_A[:,0] -= deltaNum
                x2_B[:,0] += deltaNum
                x2_C[:,1] -= deltaNum
                x2_D[:,1] += deltaNum
                p_A   = net(x1_dev,x2_A)
                p_B   = net(x1_dev,x2_B)
                p_C   = net(x1_dev,x2_C)
                p_D   = net(x1_dev,x2_D)
                ddx = (p_B-p_A) / 2.0 / deltaNum
                ddy = (p_D-p_C) / 2.0 / deltaNum

                # === uncomment 1 eikonal loss formulation ===
                # (1) mean absolute difference of squared gradient
                lossEik = torch.mean(torch.abs( ddx*ddx+ddy*ddy - 1.0 ))
                # (2) L2 loss of gradient difference
                # lossEik = torch.mean(torch.square( torch.sqrt(ddx*ddx+ddy*ddy) - 1.0 ))
            else:
                lossEik = 0.0

            loss = lossWL1 + lossEik*eikonalFactor
            loss.backward()

            abs_err = torch.abs(p_dev-y_dev)
            tt_loss_tr_ep[iMB_] = abs_err.to('cpu')
            lossWL1TrainE += lossWL1.item()
            if useEik:
                lossEikTrainE += lossEik.item()*eikonalFactor
            lossTrainE    += loss.item()

        optimizer.step()


        # lossTrainE   += loss.item()
        # if iB % printEvery == printEvery-1:
        #     #print('x',end='',flush=True)'
        #     print(epoch, " ", lossTrainE/(iMB + 1))


    t_2 = time.time()
    #print('|{:06.1f}|'.format(t_2-t_1),end='',flush=True)
    net.eval()
    lossWL1TestE = 0.0
    lossEikTestE = 0.0
    lossTestE    = 0.0
    for iB in range(nBperEte):
        for iMB in range(nMBperB):
            iMB_ = range(iB*batchSize+iMB*miniBatchSize,iB*batchSize+(iMB+1)*miniBatchSize)

            x1_dev = tt_x1_tr_ep[iMB_].to(device)
            x2_dev = tt_x2_tr_ep[iMB_].to(device)
            y_dev  = tt_y_tr_ep[iMB_].to(device)
            p_dev = net(x1_dev,x2_dev)

            weights      = torch.ones_like(y_dev)
            weights_sum0 = weights.sum()
            if useWL1:
                # weights      = torch.exp(-torch.abs(y_dev)/WL1Width)
                wcos = torch.cos(torch.abs(y_dev)/WL1Width*2.0/torch.pi)
                wcos[torch.abs(y_dev)>2.0*WL1Width*torch.pi] = -1.0
                wexp = torch.exp(-torch.abs(y_dev)/WL1Width)
                wsgm = torch.sigmoid((torch.abs(y_dev)-2.0*WL1Width)*3.0/WL1Width)
                weights = (1.0-wsgm)*wcos + wsgm*wexp
            weights_sum1 = weights.sum()
            weights      = weights*weights_sum0/weights_sum1

            lossWL1 = criterion(p_dev,y_dev)
            lossWL1 = lossWL1*weights
            lossWL1 = lossWL1.sum() / weights.sum()

            if useEik:
                x2_A = x2_dev.clone().detach()
                x2_B = x2_dev.clone().detach()
                x2_C = x2_dev.clone().detach()
                x2_D = x2_dev.clone().detach()
                # x2_A = x2_dev.clone()
                # x2_B = x2_dev.clone()
                # x2_C = x2_dev.clone()
                # x2_D = x2_dev.clone()
                x2_A[:,0] -= deltaNum
                x2_B[:,0] += deltaNum
                x2_C[:,1] -= deltaNum
                x2_D[:,1] += deltaNum
                p_A   = net(x1_dev,x2_A)
                p_B   = net(x1_dev,x2_B)
                p_C   = net(x1_dev,x2_C)
                p_D   = net(x1_dev,x2_D)
                ddx = (p_B-p_A) / 2.0 / deltaNum
                ddy = (p_D-p_C) / 2.0 / deltaNum

                lossEik = torch.mean(torch.abs( ddx*ddx+ddy*ddy - 1.0 ))                 # other loss
                # lossEik = torch.mean(torch.square( torch.sqrt(ddx*ddx+ddy*ddy) - 1.0 ))  # L2 loss
            else:
                lossEik = 0.0

            loss = lossWL1 + lossEik*eikonalFactor

            lossWL1TestE += lossWL1.item()
            if useEik:
                lossEikTestE += lossEik.item()*eikonalFactor
            lossTestE    += loss.item()

        # if iB % printEvery == printEvery-1:
        #     print('o',end='',flush=True)

    lossWL1TrainE /= nBperEtr
    lossEikTrainE /= nBperEtr
    lossTrainE    /= nBperEtr
    lossWL1TestE  /= nBperEte
    lossEikTestE  /= nBperEte
    lossTestE     /= nBperEte
    epochCount += 1

    lossWL1Train.append(lossWL1TrainE)
    lossEikTrain.append(lossEikTrainE)
    lossTrain.append(lossTrainE)
    lossWL1Test.append(lossWL1TestE)
    lossEikTest.append(lossEikTestE)
    lossTest.append(lossTestE)
    learnRate.append(leRa)
    epochID.append(epochCount)

    if saveEvery > 0 and ((epochCount-1) % saveEvery == saveEvery-1):
        state = {
            'lossWL1Train': lossWL1Train,
            'lossEikTrain': lossEikTrain,
            'lossTrain': lossTrain,
            'lossWL1Test': lossWL1Test,
            'lossEikTest': lossEikTest,
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
        seName = cName + '_epo{:03.0f}'.format(epochCount)
        torch.save(state,savePath+seName+'_params.pth')

    t_3 = time.time()
    print('|{:05.1f}]  TR {:8.6f} {:8.6f} {:8.6f}  TE {:8.6f} {:8.6f} {:8.6f}  {:5.1f} {:7.1f}'.format(
        t_3-t_2,lossWL1TrainE,lossEikTrainE,lossTrainE,lossWL1TestE,lossEikTestE,lossTestE,t_3-t_0,t_3-t_00),flush=True)


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
