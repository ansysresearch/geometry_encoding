for param_group in optimizer.param_groups:
    param_group['lr'] = leRa
epochCount = len(epochID)
printHeader(n_print_tr,n_print_te)
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

    print('[{:04.1f}|'.format(t_1-t_0),end='',flush=True)
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
        if iB % printEvery == printEvery-1:
            print('x',end='',flush=True)


    t_2 = time.time()
    print('|{:06.1f}|'.format(t_2-t_1),end='',flush=True)
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

            # x1_dev = tt_x1_tr_ep[iMB_].to(device)
            # x2_dev = tt_x2_tr_ep[iMB_].to(device)
            # y_dev  = tt_y_tr_ep[iMB_].to(device)
            # x2_A = x2_dev.clone().detach()
            # x2_B = x2_dev.clone().detach()
            # x2_C = x2_dev.clone().detach()
            # x2_D = x2_dev.clone().detach()
            # x2_A[:,0] -= deltaNum
            # x2_B[:,0] += deltaNum
            # x2_C[:,1] -= deltaNum
            # x2_D[:,1] += deltaNum
            #
            # p_dev = net(x1_dev,x2_dev)
            # p_A   = net(x1_dev,x2_A)
            # p_B   = net(x1_dev,x2_B)
            # p_C   = net(x1_dev,x2_C)
            # p_D   = net(x1_dev,x2_D)
            #
            # weights      = torch.ones_like(y_dev)
            # weights_sum0 = weights.sum()
            # weights      = torch.exp(-torch.abs(y_dev)/L)
            # weights_sum1 = weights.sum()
            # weights      = weights*weights_sum0/weights_sum1
            # ddx = (p_B-p_A) / 2.0 / deltaNum
            # ddy = (p_D-p_C) / 2.0 / deltaNum
            #
            # lossWL1 = criterion(p_dev,y_dev)
            # lossWL1 = lossWL1*weights
            # lossWL1 = lossWL1.sum() / weights.sum()
            # lossEik = torch.mean(torch.abs( ddx*ddx+ddy*ddy - 1.0 ))
            #
            # loss = lossWL1 + lossEik*a_lder

            lossWL1TestE += lossWL1.item()
            if useEik:
                lossEikTestE += lossEik.item()*eikonalFactor
            lossTestE    += loss.item()


            # t_A0 = time.time()
            # for i in range(batchSize):
            #     derivative = torch.autograd.grad(p_dev[i],[x2_dev],grad_outputs=None,create_graph=True)[0][i]
            #     lossEikonalA += derivative[0]+derivative[1]
            #     # print('[{:03.0f}][{:03.0f}][{:03.0f}] : {:11.8f}/{:11.8f}'.format(iB,iMB,i,derivative[0].item(),derivative[1].item()))
            #     # assert(False)
            # t_A1 = time.time()
            # dt_A += t_A1-t_A0
            # # i = 2
            # # derivative = torch.autograd.grad(p_dev[i],[x2_dev],grad_outputs=None,create_graph=True)[0][i]
            # # print('')
            # # print('derivativeATG[{:03.0f}][{:03.0f}] = {:11.8f}/{:11.8f}'.format(iB,i,derivative[0],derivative[1]),flush=True)

            # x2_dev.requires_grad = False

            # B
            #   time : 103.7 sec
            #   val  :  23.9273
            # B opt
            #   time :   2.8 sec
            #   val  :  23.9273


            # deltaNum = 1.0e-10
            # t_B0 = time.time()
            # x2_A = x2_dev.clone().detach()
            # x2_B = x2_dev.clone().detach()
            # x2_C = x2_dev.clone().detach()
            # x2_D = x2_dev.clone().detach()
            # # print('x2_A.size() = {:}'.format(x2_A.size()),flush=True)
            # x2_A[:,0] -= deltaNum
            # x2_B[:,0] += deltaNum
            # x2_C[:,1] -= deltaNum
            # x2_D[:,1] += deltaNum
            # p_A = net(x1_dev,x2_A)
            # p_B = net(x1_dev,x2_B)
            # p_C = net(x1_dev,x2_C)
            # p_D = net(x1_dev,x2_D)
            # i = 0
            # # print('A : {:10.8f}/{:10.8f} > {:11.8f}'.format(x2_A[i][0],x2_A[i][1],p_A[i].item()),flush=True)
            # # print('B : {:10.8f}/{:10.8f} > {:11.8f}'.format(x2_B[i][0],x2_B[i][1],p_B[i].item()),flush=True)
            # # print('C : {:10.8f}/{:10.8f} > {:11.8f}'.format(x2_C[i][0],x2_C[i][1],p_C[i].item()),flush=True)
            # # print('D : {:10.8f}/{:10.8f} > {:11.8f}'.format(x2_D[i][0],x2_D[i][1],p_D[i].item()),flush=True)
            # ddx = (p_B-p_A) / 2.0 / deltaNum
            # ddy = (p_D-p_C) / 2.0 / deltaNum
            # # lossE = torch.abs( ddx*ddx + ddy*ddy - 1.0 )
            # # lossEikonalB += torch.sum(ddx)+torch.sum(ddy)
            # lossEikonalB += torch.mean(torch.abs( ddx*ddx+ddy*ddy - 1.0 ))
            # # print('[{:03.0f}][{:03.0f}][{:03.0f}] :  {:11.8f}/{:11.8f}  {:10.8f}'.format(iB,iMB,i,ddx[i].item(),ddy[i].item(),lossE[i].item()),flush=True)
            # # # print('[{:03.0f}][{:03.0f}][{:03.0f}] :  {:11.8f}/{:11.8f}'.format(iB,iMB,i,ddx[i].item(),ddy[i].item()),flush=True)
            # # # print('lossE : {:}'.format(lossE),flush=True)
            # # assert(False)
            # t_B1 = time.time()
            # dt_B += t_B1-t_B0


            #  -0.00405868 / -0.00165910
            #  -0.00405877 / -0.00165902
            #  -0.00405877 / -0.00165874


            # d = 1.0e-10
            # t_B0 = time.time()
            # for i in range(batchSize):
            #     x1_der = torch.reshape(x1_dev[i].clone().detach(),(1,1,n_res_x,n_res_y)).expand(4,1,    n_res_x,n_res_y)
            #     x2_der = x2_dev[i].clone().detach().repeat(4,1)
            #     x2_der[0][0] -= d
            #     x2_der[1][0] += d
            #     x2_der[2][1] -= d
            #     x2_der[3][1] += d
            #     p_der = net(x1_der,x2_der)
            #     ddx = (p_der[1]-p_der[0])/2.0/d
            #     ddy = (p_der[3]-p_der[2])/2.0/d
            #     lossEikonalB += ddx+ddy
            #     print('[{:03.0f}][{:03.0f}][{:03.0f}] : {:11.8f}/{:11.8f}'.format(iB,iMB,i,ddx.item(),ddy.item()))
            #     assert(False)
            #     # lossEikonalB += ddx.item()+ddy.item()
            # t_B1 = time.time()
            # dt_B += t_B1-t_B0



            # p_der = net(x1_der,x2_der)
            # # print('p_der = {:}'.format(p_der),flush=True)
            # # print('p_der.size() = {:}'.format(p_der.size()),flush=True)
            # ddx = (p_der[1]-p_der[0])/2.0/d
            # ddy = (p_der[3]-p_der[2])/2.0/d
            # # print(ddx)
            # # print(ddy)
            # print('derivativeNUM[{:03.0f}][{:03.0f}] = {:11.8f}/{:11.8f}'.format(iB,i,ddx.item(),ddy.item()),flush=True)



        if iB % printEvery == printEvery-1:
            print('o',end='',flush=True)

    # print('')
    # # print('A : {:10.8f}'.format(dt_A))
    # # print('  : {:10.8f}'.format(lossEikonalA.item()))
    # print('B : {:10.8f}'.format(dt_B))
    # print('  : {:10.8f}'.format(lossEikonalB.item()))
    # # print(lossEikonalA,flush=True)
    # print(lossEikonalB,flush=True)
    #
    # assert(False)

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
