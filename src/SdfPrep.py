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
