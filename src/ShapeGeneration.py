# functions used fo shape generation


# intersectionYes = checkIntersection(i,j,xy_,order_)
def checkIntersection(i,j,xy_,order_):
    xyi1 = xy_[order_[i  ],:]
    xyi2 = xy_[order_[i+1],:]
    xyj1 = xy_[order_[j  ],:]
    xyj2 = xy_[order_[j+1],:]

    ni = np.array([xyi1[1]-xyi2[1],xyi2[0]-xyi1[0]])
    nj = np.array([xyj1[1]-xyj2[1],xyj2[0]-xyj1[0]])

    if (np.dot(xyi1-xyj1,nj) * np.dot(xyi2-xyj1,nj) < 0) and (np.dot(xyj1-xyi1,ni) * np.dot(xyj2-xyi1,ni) < 0):
        intersectionYes = True
    else:
        intersectionYes = False

    return intersectionYes



# orderNew_ = fixIntersection(i,j,order_)
def  fixIntersection(i,j,order_):
    orderNew_ = order_.copy()

    for k in range(j-i):
        orderNew_[i+k+1] = order_[j-k]

    return orderNew_


# order_,intersectionCount = checkAndFixIntersection(xy_,order_)
#  - checkIntersection
#  - fixIntersection
def checkAndFixIntersection(xy_,order_):
    N    = xy_.shape[0]
    Nmax = np.max([100,N])
    Nmax = np.max([1,N])

    intersectionCount = 0
    whileCount        = 0
    foundIntersection = True

    while foundIntersection:
        foundIntersection = False
        whileCount += 1
        for i in range(N):
            for j in range(i+2,N):
                intersectionYes = checkIntersection(i,j,xy_,order_)
                if intersectionYes:
                    order_ = fixIntersection(i,j,order_)
                    intersectionCount += 1
                foundIntersection = foundIntersection or intersectionYes
        if whileCount > Nmax and foundIntersection:
            print('checkAndFixIntersection::whileCount = {:}.'.format(whileCount))
            return order_,-1
            assert(False)

    return order_,intersectionCount


# xy_, moveCount = adjustCtrlPts(xy_,order_)
def adjustCtrlPts(xy_,order_):
    N    = xy_.shape[0]
    Nmax = np.max([100,N])
    Nmax = np.max([1,N])

    moveCount  = 0
    whileCount = 0
    movedPt    = True

    delta = 1e-3
    dist  = np.min([0.1, 1.0/N])

    while movedPt:
        whileCount += 1
        movedPt     = False
        for i in range(N):
            xyi = xy_[order_[i],:]
            for j in range(N):
                if j==i or j==i-1 or j==i-1+N:
                    continue
                xy1 = xy_[order_[j  ],:]
                xy2 = xy_[order_[j+1],:]
                npts = int(np.ceil(np.linalg.norm(xy1-xy2)/delta))
                lpts = np.concatenate([np.reshape(np.linspace(xy1[0],xy2[0],npts),(1,-1)),
                                       np.reshape(np.linspace(xy1[1],xy2[1],npts),(1,-1))],axis=0)
                d = np.sqrt(np.sum((lpts.transpose()-xyi)**2,axis=1))
                dmin = d.min()
                if dmin < dist:
                    n1 = xy1-xy2
                    n2 = np.array([xy1[1]-xy2[1],xy2[0]-xy1[0]])
                    n1 /= np.linalg.norm(n1)
                    n2 /= np.linalg.norm(n2)

                    td1 = np.dot(xy1,n1)
                    td2 = np.dot(xy2,n1)
                    tdi = np.dot(xyi,n1)

                    nd1 = np.dot(xy1,n2)
                    ndi = np.dot(xyi-xy1,n2)

                    if (tdi-td1)*(tdi-td2) > 0:
                        dmin  = np.abs([tdi-td1,tdi-td2]).min()
                        dMove = np.sqrt(dist**2-dmin**2) #- np.abs(ndi)
                    else:
                        dMove = dist

                    newXyi = tdi*n1 + (nd1+np.sign(ndi)*dMove)*n2
                    xy_[order_[i],:] = newXyi
                    movedPt = True
                    moveCount += 1
        if whileCount > Nmax and movedPt:
            print('adjustCtrlPts::whileCount = {:}.'.format(whileCount))
            return xy_,-1

    return xy_, moveCount



# plotPolygon(xy_,order_)
def plotPolygon(xy_,order_):
    plt.figure()
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.plot(xy_[order_,0],xy_[order_,1],'-s')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()


# xy_,order_ = sampleRaw(N)
def sampleRaw(N):
    xy_ = np.random.uniform(0.0, 1.0, (N, 2))
    order_ = np.random.permutation(N)
    order_ = np.append(order_,[order_[0]],axis=0)
    return xy_,order_



# xy_,order_ = samplePolygon(N,showProg)
def samplePolygon(N,showProg):
    xy_,order_ = sampleRaw(N)

    if showProg:
        plotPolygon(xy_,order_)

    intersectionCount = 1
    moveCount         = 1
    whileCount        = 0

    if showProg:
        print('begin processing')
    while intersectionCount or moveCount:
        order_, intersectionCount = checkAndFixIntersection(xy_,order_)
        if intersectionCount==-1:
            if showProg:
                print('intersectionCount::resampling')
            xy_,order_ = sampleRaw(N)
            if showProg:
                plotPolygon(xy_,order_)
            intersectionCount = 1
            moveCount         = 1
            whileCount        = 0
            continue

        if intersectionCount and showProg:
            plotPolygon(xy_,order_)

        xy_,    moveCount         = adjustCtrlPts(xy_,order_)
        if moveCount==-1:
            if showProg:
                print('moveCount::resampling')
            xy_,order_ = sampleRaw(N)
            if showProg:
                plotPolygon(xy_,order_)
            intersectionCount = 1
            moveCount         = 1
            whileCount        = 0
            continue

        if moveCount and showProg:
            plotPolygon(xy_,order_)

        whileCount += 1
        if whileCount>=100:
            if showProg:
                print('whileCount::resampling')
            xy_,order_ = sampleRaw(N)
            if showProg:
                plotPolygon(xy_,order_)
            intersectionCount = 1
            moveCount         = 1
            whileCount        = 0
            continue

        if showProg:
            print('iteration # {:}\t# of intersections : {:}\t# of moves : {:}'.format(whileCount,intersectionCount,moveCount))

    return xy_,order_



# P_i = deBoor(k,i,xi,Xi_,P_,p):
#  - deBoor
def deBoor(k,i,xi,Xi_,P_,p):
    if k==1:
        return P_[i,:]
    alpha_ki = (xi-Xi_[i]) / (Xi_[i+p+2-k]-Xi_[i])
    if Xi_[i+p+2-k]==Xi_[i]:
        alpha_ki = 0.0
    return (1.0-alpha_ki)*deBoor(k-1,i-1,xi,Xi_,P_,p) + alpha_ki*deBoor(k-1,i,xi,Xi_,P_,p)



# P_i = splineCurve(xi,Xi_,P_,p)
#  - deBoor
def splineCurve(xi,Xi_,P_,p):
    if xi == Xi_[-1]:
        xi -= xi*np.finfo(Xi_.dtype).eps

    l = 0
    assert(Xi_[l]<=xi)
    while Xi_[l+1]<=xi:
        l += 1
    return deBoor(p+1,l,xi,Xi_,P_,p)



# plotBSpline(P_,p)
#  - splineCurve
#     - deBoor
def plotBSpline(P_,p):
    n_eval = int(1e3)

    Pc = 0.5*np.reshape(P_[0]+P_[-1],(1,-1))
    P_ = np.concatenate([Pc,P_,Pc],axis=0)
    n  = P_.shape[0]
    d  = P_.shape[1]

    Xi_ = np.concatenate([np.zeros((3,)), np.arange(n-p-1)+1, np.ones((3,))*(n-p)])

    Xi_eval = np.linspace(Xi_[0],Xi_[-1],num=n_eval)
    Xi_knot = np.unique(Xi_)
    P_eval  = np.zeros((n_eval,d))
    P_knot  = np.zeros((Xi_knot.size,d))

    for ie,xi in enumerate(Xi_eval):
        P_eval[ie] = splineCurve(xi,Xi_,P_,p)
    for ik,xi in enumerate(Xi_knot):
        P_knot[ik] = splineCurve(xi,Xi_,P_,p)


    plt.figure()
    ax = plt.gca()
    ax.set_aspect(1.0)

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    plt.plot(P_[1:-1,0],P_[1:-1,1],'sk')
    plt.plot(P_[:,0],P_[:,1],'-k')

    plt.plot(P_eval[:,0],P_eval[:,1],'-')
    plt.plot(P_knot[:,0],P_knot[:,1],'dr')

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.show()


# P_eval = getBSpline(P_,p, delta)
#  - splineCurve
#     - deBoor
def getBSpline(P_,p, delta):
    d0     = np.reshape(P_[-1]-P_[0],(1,-1))
    dP_    = np.linalg.norm(np.concatenate((d0, (P_[1:]-P_[:-1]), d0),axis=0),axis=1)
    dK_    = 0.5*(dP_[:-1] + dP_[1:])
    dKmax  = np.max(dK_)
    fos    = 2.0
    n_eval = int(np.ceil(dKmax*P_.shape[0]/delta*fos))

    Pc = 0.5*np.reshape(P_[0]+P_[-1],(1,-1))
    P_ = np.concatenate([Pc,P_,Pc],axis=0)
    n  = P_.shape[0]
    d  = P_.shape[1]

    Xi_ = np.concatenate([np.zeros((3,)), np.arange(n-p-1)+1, np.ones((3,))*(n-p)])

    Xi_eval = np.linspace(Xi_[0],Xi_[-1],num=n_eval)
    P_eval  = np.zeros((n_eval,d))

    for ie,xi in enumerate(Xi_eval):
        P_eval[ie] = splineCurve(xi,Xi_,P_,p)

    return P_eval



def sampleCircle(x,y,radius,print_sampling):
    zi   = np.zeros(x.shape)
    xy_c = np.random.uniform(0,1,2)
    r    = np.sqrt( np.square(x-xy_c[0]) + np.square(y-xy_c[1]) )
    zi[r<radius] = 1.0

    nSurfPts = int(np.ceil(2.0*np.pi*radius/delta))
    angles   = np.linspace(0.0,2.0*np.pi,nSurfPts,endpoint=False).reshape((nSurfPts,1))
    surfacePts = np.concatenate((xy_c[0]+radius*np.cos(angles), xy_c[1]+radius*np.sin(angles)), axis=1)
    shapeParams = {
        'shape':'circ',
        'center':xy_c,
        'radius':radius
    }

    if print_sampling:
        print('circle       >>  origin {:.3f},{:.3f}\t>>  radius {:.3f}'.format(xy_c[0],xy_c[1],radius))

    return shapeParams,zi,surfacePts


def sampleRectangle(x,y,L,print_sampling):
    zi        = np.zeros(x.shape)
    aspectRat = np.random.uniform(1.0,2.0)
    xy_c      = np.random.uniform(0,1,2)
    angle     = np.random.uniform(0,np.pi)

    halfD = L*np.sqrt(1.0+aspectRat)*0.5
    halfA = np.arctan(1.0/aspectRat)

    c1 = xy_c + halfD*np.array([np.cos(angle+halfA      ),np.sin(angle+halfA    )])
    c2 = xy_c + halfD*np.array([np.cos(angle-halfA+np.pi),np.sin(angle-halfA+np.pi)])
    c3 = xy_c + halfD*np.array([np.cos(angle+halfA+np.pi),np.sin(angle+halfA+np.pi)])
    c4 = xy_c + halfD*np.array([np.cos(angle-halfA      ),np.sin(angle-halfA    )])
    ci = np.array([c1,c2,c3,c4,c1])
    for ic in range(4):
        A = ci[ic+0]
        B = ci[ic+1]
        s_ = ((B[0]-A[0])*(y-A[1]) - (B[1]-A[1])*(x-A[0]))
        s = s_ < 0
        zi += s

    nSurfPtsA = int(np.ceil(L*aspectRat/delta))
    nSurfPtsB = int(np.ceil(L/delta))
    surfacePtsA1 = np.concatenate( (np.linspace(c1[0],c2[0],nSurfPtsA,endpoint=False).reshape((nSurfPtsA,1)),
                                   np.linspace(c1[1],c2[1],nSurfPtsA,endpoint=False).reshape((nSurfPtsA,1))),
                                   axis=1)
    surfacePtsB1 = np.concatenate( (np.linspace(c2[0],c3[0],nSurfPtsB,endpoint=False).reshape((nSurfPtsB,1)),
                                   np.linspace(c2[1],c3[1],nSurfPtsB,endpoint=False).reshape((nSurfPtsB,1))),
                                   axis=1)
    surfacePtsA2 = np.concatenate( (np.linspace(c3[0],c4[0],nSurfPtsA,endpoint=False).reshape((nSurfPtsA,1)),
                                   np.linspace(c3[1],c4[1],nSurfPtsA,endpoint=False).reshape((nSurfPtsA,1))),
                                   axis=1)
    surfacePtsB2 = np.concatenate( (np.linspace(c4[0],c1[0],nSurfPtsB,endpoint=False).reshape((nSurfPtsB,1)),
                                   np.linspace(c4[1],c1[1],nSurfPtsB,endpoint=False).reshape((nSurfPtsB,1))),
                                   axis=1)
    surfacePts = np.concatenate((surfacePtsA1,surfacePtsB1,surfacePtsA2,surfacePtsB2),axis=0)

    shapeParams = {
        'shape':'rect',
        'center':xy_c,
        'L':L,
        'aspectRat':aspectRat,
        'angle':angle
    }

    if print_sampling:
        print('rectangle    >>  origin {:.3f},{:.3f}\t>>  {:.3f}x{:.3f}\t>>  {:5.1f} deg'.format(
            xy_c[0],xy_c[1],L,L*aspectRat,angle*180.0/np.pi))

    return(shapeParams,zi,surfacePts)


def sampleTriangle(x,y,L1,print_sampling):
    xy_c    = np.random.uniform(0,1,2)
    phi1    = np.random.uniform(0,2.0*np.pi)
    phi2deg = np.random.uniform(minTriAngle,180.0-2.0*minTriAngle)
    phi3deg = np.random.uniform(minTriAngle,180.0-1.0*minTriAngle-phi2deg)
    phi2    = phi2deg * np.pi/180.0
    phi3    = phi3deg * np.pi/180.0

    L2 = L1 * np.sin(phi2) / np.sin(np.pi - phi2 - phi3)
    L3 = L1 * np.sin(phi3) / np.sin(np.pi - phi2 - phi3)

    c1 = xy_c
    c2 = xy_c + L1*np.array([np.cos(phi1     ),np.sin(phi1     )])
    c3 = xy_c + L3*np.array([np.cos(phi1+phi2),np.sin(phi1+phi2)])
    ci = np.array([c1,c2,c3,c1])
    zi = np.zeros(x.shape)
    for ic in range(3):
        A = ci[ic+0]
        B = ci[ic+1]
        s_ = ((B[0]-A[0])*(y-A[1]) - (B[1]-A[1])*(x-A[0]))
        s = s_ < 0
        zi += s

    nSurfPts1 = int(np.ceil(L1/delta))
    nSurfPts2 = int(np.ceil(L2/delta))
    nSurfPts3 = int(np.ceil(L3/delta))
    surfacePts1 = np.concatenate( (np.linspace(c1[0],c2[0],nSurfPts1,endpoint=False).reshape((nSurfPts1,1)),
                                   np.linspace(c1[1],c2[1],nSurfPts1,endpoint=False).reshape((nSurfPts1,1))),
                                   axis=1)
    surfacePts2 = np.concatenate( (np.linspace(c2[0],c3[0],nSurfPts2,endpoint=False).reshape((nSurfPts2,1)),
                                   np.linspace(c2[1],c3[1],nSurfPts2,endpoint=False).reshape((nSurfPts2,1))),
                                   axis=1)
    surfacePts3 = np.concatenate( (np.linspace(c3[0],c1[0],nSurfPts3,endpoint=False).reshape((nSurfPts3,1)),
                                   np.linspace(c3[1],c1[1],nSurfPts3,endpoint=False).reshape((nSurfPts3,1))),
                                   axis=1)
    surfacePts = np.concatenate((surfacePts1,surfacePts2,surfacePts3),axis=0)

    shapeParams = {
        'shape':'tria',
        'center':xy_c,
        'L1':L1,
        'phi1':phi1,
        'phi2':phi2,
        'phi3':phi3,
    }

    if print_sampling:
        print('triangle     >>  origin {:.3f},{:.3f}\t>>  {:.3f},{:.3f},{:.3f}\t>>  {:5.1f} deg'.format(
            xy_c[0],xy_c[1],L1,L2,L3,phi1*180.0/np.pi))

    return shapeParams,zi,surfacePts


def sampleBSpline(x,y,L,print_sampling):
    n_pts = np.random.randint(7,21)
    xy_c  = np.random.uniform(0,1,2)-0.5

    xy_,order_ = samplePolygon(n_pts,False)

    cwSum = 0.0
    for i in range(n_pts):
        x1 = xy_[order_[i  ],0]
        y1 = xy_[order_[i  ],1]
        x2 = xy_[order_[i+1],0]
        y2 = xy_[order_[i+1],1]
        cwSum += (x2-x1) / (y2+y1)
    inOrOut = np.sign(cwSum)

    if cwSum>0:
        cwS = 'ccw'
    else:
        cwS = 'cw'

    xy_ = (xy_-xy_c)
    P = xy_[order_[:-1],:]
    surfacePts = getBSpline(P,2,delta)

    xyi = np.zeros((1,2))
    zi  = np.zeros(x.shape)

    for ix in range(x.shape[1]):
        for iy in range(y.shape[0]):
            xi = x[iy,ix]
            yi = y[iy,ix]
            xyi[0,0] = xi
            xyi[0,1] = yi
            dista = np.linalg.norm(surfacePts-xyi,axis=1)
            imin = np.argmin(dista)
            dmin = np.amin(dista)
            if imin == dista.size-1:
                imin = 0

            P1 = surfacePts[imin  ,:]
            P2 = surfacePts[imin+1,:]

            s = inOrOut * np.sign((P2[0]-P1[0])*(yi-P1[1]) - (P2[1]-P1[1])*(xi-P1[0]))
            if s > 0:
                zi[iy,ix] = 1.0

    shapeParams = {
        'shape':'poly',
        'xy_':xy_,
        'order_':order_,
        'inOrOut':inOrOut,
        # 'sfpts':surfacePts
    }

    if print_sampling:
        print('polygon      >>  origin {:.3f},{:.3f}\t>>  {:.3f},{:2.0f}\t>>  {:}'.format(
            xy_c[0],xy_c[1],L,n_pts,cwS))

    return shapeParams,zi,surfacePts


# surfacePtsAll,xyd = secondCircle(shapeParams,surfacePtsAll,xyd,tol)
def secondCircle(shapeParams,surfacePtsAll,xyd,tol):
    radius = shapeParams['radius']
    xy_c   = shapeParams['center']

    surfaceRad2 = np.square(surfacePtsAll[:,0]-xy_c[0]) + np.square(surfacePtsAll[:,1]-xy_c[1])
    sampleRad2  = np.square(xyd[:,0]          -xy_c[0]) + np.square(xyd[:,1]          -xy_c[1])

    surfacePtsOut = np.sqrt( surfaceRad2 ) - radius > -tol
    samplePtsIn   = np.sqrt( sampleRad2  ) - radius < 0

    surfacePtsAll = surfacePtsAll[surfacePtsOut,:]
    xyd[samplePtsIn,2] = -1.0

    return surfacePtsAll,xyd


# surfacePtsAll,xyd = secondRectangle(shapeParams,surfacePtsAll,xyd,tol)
def secondRectangle(shapeParams,surfacePtsAll,xyd,tol):
    xy_c   = shapeParams['center']
    L      = shapeParams['L']
    aspectRat = shapeParams['aspectRat']
    angle  = shapeParams['angle']

    halfD = L*np.sqrt(1.0+aspectRat)*0.5
    halfA = np.arctan(1.0/aspectRat)

    c1 = xy_c + halfD*np.array([np.cos(angle+halfA      ),np.sin(angle+halfA    )])
    c2 = xy_c + halfD*np.array([np.cos(angle-halfA+np.pi),np.sin(angle-halfA+np.pi)])
    c3 = xy_c + halfD*np.array([np.cos(angle+halfA+np.pi),np.sin(angle+halfA+np.pi)])
    c4 = xy_c + halfD*np.array([np.cos(angle-halfA      ),np.sin(angle-halfA    )])

    ci = np.array([c1,c2,c3,c4,c1])
    surfacePtsOut = np.zeros((surfacePtsAll.shape[0],))
    samplePtsOut  = np.zeros((xyd.shape[0],))

    for ic in range(4):
        A = ci[ic+0]
        B = ci[ic+1]
        surf_   = ((B[0]-A[0])*(surfacePtsAll[:,1]-A[1]) - (B[1]-A[1])*(surfacePtsAll[:,0]-A[0]))
        sample_ = ((B[0]-A[0])*(xyd[:,1]          -A[1]) - (B[1]-A[1])*(xyd[:,0]          -A[0]))
        surfacePtsOut += surf_   < tol
        samplePtsOut  += sample_ < 0

    surfacePtsAll = surfacePtsAll[surfacePtsOut>0,:]
    xyd[samplePtsOut==0,2] = -1.0

    return surfacePtsAll,xyd


# surfacePtsAll,xyd = secondTriangle(shapeParams,surfacePtsAll,xyd,tol)
def secondTriangle(shapeParams,surfacePtsAll,xyd,tol):
    xy_c = shapeParams['center']
    L1   = shapeParams['L1']
    phi1 = shapeParams['phi1']
    phi2 = shapeParams['phi2']
    phi3 = shapeParams['phi3']

    L3 = L1 * np.sin(phi3) / np.sin(np.pi - phi2 - phi3)

    c1 = xy_c
    c2 = xy_c + L1*np.array([np.cos(phi1     ),np.sin(phi1     )])
    c3 = xy_c + L3*np.array([np.cos(phi1+phi2),np.sin(phi1+phi2)])
    ci = np.array([c1,c2,c3,c1])

    surfacePtsOut = np.zeros((surfacePtsAll.shape[0],))
    samplePtsOut  = np.zeros((xyd.shape[0],))

    for ic in range(3):
        A = ci[ic+0]
        B = ci[ic+1]
        surf_   = ((B[0]-A[0])*(surfacePtsAll[:,1]-A[1]) - (B[1]-A[1])*(surfacePtsAll[:,0]-A[0]))
        sample_ = ((B[0]-A[0])*(xyd[:,1]          -A[1]) - (B[1]-A[1])*(xyd[:,0]          -A[0]))
        surfacePtsOut += surf_   < tol
        samplePtsOut  += sample_ < 0

    surfacePtsAll = surfacePtsAll[surfacePtsOut>0,:]
    xyd[samplePtsOut==0,2] = -1.0

    return surfacePtsAll,xyd


# surfacePtsAll,xyd = secondBSpline(shapeParams,surfacePtsAll,xyd,tol)
# - getBSpline
#    - splineCurve
#       - deBoor
def secondBSpline(shapeParams,surfacePtsAll,xyd,tol):
    xy_     = shapeParams['xy_']
    order_  = shapeParams['order_']
    inOrOut = shapeParams['inOrOut']
    # surfacePts = shapeParams['sfpts']

    n_pts = xy_.shape[0]
    N_loc = xyd.shape[0]

    P = xy_[order_[:-1],:]
    surfacePts = getBSpline(P,2,delta)

    xyi = np.zeros((1,2))

    i_ = np.zeros((surfacePtsAll.shape[0],))

    for i in range(surfacePtsAll.shape[0]):
        dista = np.linalg.norm(surfacePtsAll[i,:]-surfacePts,axis=1)
        imin = np.argmin(dista)
        dmin = np.amin(dista)
        if imin == dista.size-1:
            imin = 0

        P1 = surfacePts[imin  ,:]
        P2 = surfacePts[imin+1,:]

        n  = np.array([P1[1]-P2[1],P2[0]-P1[0]])
        n /= np.linalg.norm(n)

        ds = np.dot(surfacePtsAll[i,:]-P1,n)*inOrOut
        if ds < tol:
            i_[i] = 1.0
    surfacePtsAll = surfacePtsAll[i_==1.0,:]

    for ixy in range(N_loc):
        xi = xyd[ixy,0]
        yi = xyd[ixy,1]
        xyi[0,0] = xi
        xyi[0,1] = yi
        dista = np.linalg.norm(surfacePts-xyi,axis=1)
        imin = np.argmin(dista)
        dmin = np.amin(dista)
        if imin == dista.size-1:
            imin = 0

        P1 = surfacePts[imin  ,:]
        P2 = surfacePts[imin+1,:]

        zi = inOrOut * np.sign((P2[0]-P1[0])*(yi-P1[1]) - (P2[1]-P1[1])*(xi-P1[0]))
        if zi > 0:
            xyd[ixy,2] = -1.0

    return surfacePtsAll,xyd


# surfacePtsAll,xyd = secondLoop(shapeParams,surfacePtsAll,xyd,tol)
#  - secondCircle
#  - secondRectangle
#  - secondTriangle
#  - secondBSpline
#     - getBSpline
#        - splineCurve
#           - deBoor
def secondLoop(shapeParams,surfacePtsAll,xyd,tol):
    if   shapeParams['shape']=='circ':
        surfacePtsAll,xyd = secondCircle(   shapeParams,surfacePtsAll,xyd,tol)
    elif shapeParams['shape']=='rect':
        surfacePtsAll,xyd = secondRectangle(shapeParams,surfacePtsAll,xyd,tol)
    elif shapeParams['shape']=='tria':
        surfacePtsAll,xyd = secondTriangle( shapeParams,surfacePtsAll,xyd,tol)
    elif shapeParams['shape']=='poly':
        surfacePtsAll,xyd = secondBSpline(  shapeParams,surfacePtsAll,xyd,tol)
    else:
        assert(False)
    return surfacePtsAll,xyd



# z,xyd = createSampleIllu(x,y,z,L_ref,n_shape,n_samp,n_ep_dat,x_illu,y_illu)
#   - getBSpline
#      - splineCurve
#         - deBoor
def createSampleIllu(x,y,z,L_ref,n_shape,n_samp,n_ep_dat,x_illu,y_illu):
    l_ref = L_ref / np.sqrt(n_shape)
    xyd = np.concatenate( (x_illu, y_illu, np.ones((n_samp*n_ep_dat,1))), axis=1 )

    shapeList = []
    surfacePtsAll = np.zeros((0,2))
    z[:,:] = -1.0


    # first loop:
    # - sample shapes
    # - compute indicator map z
    # - compute surface points surfacePtsAll
    # - remember shape parameters shapeParams

    # plt.figure()

    for i_shape in range(int(n_shape)):
        i_rand = np.random.randint(4)
        l = max(0.1*l_ref,np.random.normal(l_ref,0.2*l_ref))

        if i_rand==0:    # circle
            shapeParams, zi, surfacePts = sampleCircle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==1.0] = 1.0
            surfacePtsAll = np.concatenate((surfacePtsAll,surfacePts),axis=0)

        elif i_rand==1:  # rectangle
            shapeParams, zi, surfacePts = sampleRectangle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==0.0] = 1.0
            surfacePtsAll = np.concatenate((surfacePtsAll,surfacePts),axis=0)

        elif i_rand==2:  # triangle
            shapeParams, zi, surfacePts = sampleTriangle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==0.0] = 1.0
            surfacePtsAll = np.concatenate((surfacePtsAll,surfacePts),axis=0)

        elif i_rand==3:  # polygon
            shapeParams, zi, surfacePts = sampleBSpline(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==1.0] = 1.0
            surfacePtsAll = np.concatenate((surfacePtsAll,surfacePts),axis=0)

    surfacePtsIn = np.logical_and.reduce(
        (surfacePtsAll[:,0]>=0, surfacePtsAll[:,0]<=1, surfacePtsAll[:,1]>=0, surfacePtsAll[:,1]<=1))
    surfacePtsAll = surfacePtsAll[surfacePtsIn,:]


    # second loop
    # - check if any surfacePts inside of shape and remove accordingly
    # - set sdf prep value for interior points to -1.0

    for i_shape in range(int(n_shape)):
        shapeParams = shapeList[i_shape]
        surfacePtsAll,xyd = secondLoop(shapeParams,surfacePtsAll,xyd,tol)


    # compute distances for sample locations

    for i in range(n_samp*n_ep_dat):
        xyd[i,2] *= np.min( np.sqrt(np.square(xyd[i,0]-surfacePtsAll[:,0]) + np.square(xyd[i,1]-surfacePtsAll[:,1]) ) )


    return z, xyd



def createSampleBinClass(x,y,z,L_ref,n_shape):
    l_ref = L_ref / np.sqrt(n_shape)

    shapeList = []
    surfacePtsAll = np.zeros((0,2))
    z[:,:] = -1.0


    # first loop:
    # - sample shapes
    # - compute indicator map z
    # - compute surface points surfacePtsAll
    # - remember shape parameters shapeParams

    # plt.figure()

    for i_shape in range(int(n_shape)):
        i_rand = np.random.randint(4)
        l = max(0.1*l_ref,np.random.normal(l_ref,0.2*l_ref))

        if i_rand==0:    # circle
            shapeParams, zi, surfacePts = sampleCircle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==1.0] = 1.0

        elif i_rand==1:  # rectangle
            shapeParams, zi, surfacePts = sampleRectangle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==0.0] = 1.0

        elif i_rand==2:  # triangle
            shapeParams, zi, surfacePts = sampleTriangle(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==0.0] = 1.0

        elif i_rand==3:  # polygon
            shapeParams, zi, surfacePts = sampleBSpline(x,y,l,print_sampling)
            shapeList.append(shapeParams)
            z[zi==1.0] = 1.0

    return z


# z,xyd = createSample(x,y,z,L_ref,n_shape,n_samp,n_ep_dat)
#  - createSampleIllu
#     - getBSpline
#        - splineCurve
#           - deBoor
def createSample(x,y,z,L_ref,n_shape,n_samp,n_ep_dat):
    x_illu = np.random.uniform(0,1,(n_samp*n_ep_dat,1))
    y_illu = np.random.uniform(0,1,(n_samp*n_ep_dat,1))
    return createSampleIllu(x,y,z,L_ref,n_shape,n_samp,n_ep_dat,x_illu,y_illu)
