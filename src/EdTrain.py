for param_group in optimizer.param_groups:
    param_group['lr'] = leRa
epochCount = len(epochID)
printHeader(n_print_tr,n_print_te)
t_00 = time.time()


for epoch in range(nEpochs):
    print('{:04.0f}\t'.format(epochCount+1),end='',flush=True)

    t_0 = time.time()

    t_1 = time.time()

    print('[{:04.1f}|'.format(t_1-t_0),end='',flush=True)
    net.train()
    lossTrainE   = 0.0
    for iB in range(nBperEtr):
        optimizer.zero_grad()

        for iMB in range(nMBperB):
            iMB_ = range(iB*batchSize+iMB*miniBatchSize,iB*batchSize+(iMB+1)*miniBatchSize)
            x1_dev = tt_x1_tr[iMB_].to(device)
            y1_dev = tt_x1_tr[iMB_].to(device)
            y1_dev[y1_dev==-1.0] = 0.0

            p_dev = net(x1_dev)
            loss  = criterion(p_dev,y1_dev)
            loss.backward()

            lossTrainE += loss.item()

        optimizer.step()

        if iB % printEvery == printEvery-1:
            print('x',end='',flush=True)


    t_2 = time.time()
    print('|{:06.1f}|'.format(t_2-t_1),end='',flush=True)
    net.eval()
    lossTestE = 0.0
    for iB in range(nBperEte):
        for iMB in range(nMBperB):
            iMB_ = range(iB*batchSize+iMB*miniBatchSize,iB*batchSize+(iMB+1)*miniBatchSize)
            x1_dev = tt_x1_te[iMB_].to(device)
            y1_dev = tt_x1_te[iMB_].to(device)
            y1_dev[y1_dev==-1.0] = 0.0

            p_dev = net(x1_dev)
            loss  = criterion(p_dev,y1_dev)

            lossTestE   += loss.item()

        if iB % printEvery == printEvery-1:
            print('o',end='',flush=True)


    lossTrainE /= nBperEtr
    lossTestE  /= nBperEte
    epochCount += 1

    lossTrain.append(lossTrainE)
    lossTest.append(lossTestE)
    learnRate.append(leRa)
    epochID.append(epochCount)

    if saveEvery > 0 and ((epochCount-1) % saveEvery == saveEvery-1):
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
        seName = cName + '_epo{:03.0f}'.format(epochCount)
        torch.save(state,savePath+seName+'_params.pth')

    t_3 = time.time()
    print('|{:05.1f}]\t{:10.8f}\t{:10.8f}\t{:5.1f}\t{:7.1f}'.format(t_3-t_2,lossTrainE,lossTestE,t_3-t_0,t_3-t_00),flush=True)
