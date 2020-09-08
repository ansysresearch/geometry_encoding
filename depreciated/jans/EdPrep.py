X1[X1==0.0] = -1.0

print('')
print('X1  :  ({:4.0f},{:3.0f},{:3.0f})    '.format(
    X1.shape[0] ,X1.shape[1] ,X1.shape[2]),end='')
print('in [{:7.4f},{:7.4f},{:7.4f}]'.format(
    np.min(X1) ,np.mean(X1) ,np.max(X1)))
print('')


print('neural network parameters')
printParamNumber(net)
print('')


frac_train_eff = frac_train*(1+frac_rsmpl) / (frac_train*(1+frac_rsmpl)+(1-frac_train))

n_geo      = int(X1.shape[0])
n_res_x    = int(X1.shape[1])
n_res_y    = int(X1.shape[2])

n_geo_tr = int(frac_train*X1.shape[0])
n_geo_te = n_geo - n_geo_tr
n_res_geo_tr = int(n_geo_tr*frac_rsmpl)
n_tot_geo_tr = n_geo_tr + n_res_geo_tr

nBperEtr   = int(n_geo_tr/batchSize)
nBperEte   = int(n_geo_te/batchSize)
nMBperB    = int(batchSize/miniBatchSize)
n_print_tr = int(nBperEtr/printEvery)
n_print_te = int(nBperEte/printEvery)


print('data set parameters')
print('  # of geometries              : {:5.0f}'.format(n_geo))
print('  image resolution             : {:2.0f}x{:2.0f}'.format(n_res_x,n_res_y))
print('')

print('training parameters')
print('  training fraction : {:5.3f}'.format(frac_train))
print('    # of training geo\'s : {:5.0f}'.format(n_geo_tr))
print('    # of testing geo\'s  : {:5.0f}'.format(n_geo_te))
print('  resample fraction : {:5.3f}'.format(frac_rsmpl))
print('    effective training fraction               : {:6.3f}'.format(frac_train_eff))
print('    # of resampled training samples per epoch : {:6.0f}'.format(n_res_geo_tr))
print('    total # of training samples per epoch     : {:6.0f}'.format(n_tot_geo_tr))
print('  (mini) batch size : {:3.0f}  ({:3.0f})'.format(batchSize,miniBatchSize))
print('    # of training batches per epoch : {:4.0f}'.format(nBperEtr))
print('    # of testing batches per epoch  : {:4.0f}'.format(nBperEte))
print('    # of mini batches per batch     : {:4.0f}'.format(nMBperB))
print('  print every {:} batches'.format(printEvery))
print('    # of training prints : {:3.0f}'.format(n_print_tr))
print('    # of testing prints  : {:3.0f}'.format(n_print_te),flush=True)


tt_loss_tr_ep = torch.from_numpy(np.zeros((n_tot_geo_tr,1)))

tt_x1_tr = torch.from_numpy(np.reshape(X1[:n_geo_tr],(n_geo_tr,1,n_res_x,n_res_y)))
tt_x1_te = torch.from_numpy(np.reshape(X1[n_geo_tr:],(n_geo_te,1,n_res_x,n_res_y)))

x1_dev = torch.from_numpy(np.zeros((miniBatchSize,1,n_res_x,n_res_y))).to(device)


resolutionID = n_res_x

cName = 'img{:03.0f}_arc{:02.0f}_dat{:02.0f}_{:}'.format(resolutionID,networkID,datasetID,caseName)
print('\n\nThis case\'s name is :  ' + cName,flush=True)

if startNew:
    print('Starting new')
    lossTrain = []
    lossTest  = []
    learnRate = []
    epochID   = []
    np.random.seed(1234)
else:
    seName = cName + '_epo{:03.0f}'.format(oldEpoch)
    print('Starting from epoch {:}'.format(oldEpoch))
    state = torch.load(savePath+seName+'_params.pth')
    lossTrain     = state['lossTrain']
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


iShuffle = np.random.permutation(n_geo_tr)
tt_x1_tr = tt_x1_tr[iShuffle]
iShuffle = np.random.permutation(n_geo_te)
tt_x1_te = tt_x1_te[iShuffle]


def printHeader(n_print_tr,n_print_te):
    print('    \t[    |',end='')
    for i in range(n_print_tr):
        print(' ',end='')
    print('|      |',end='')
    for i in range(n_print_te):
        print(' ',end='')
    print('|     ]')
