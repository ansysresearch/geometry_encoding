import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import time
import statistics as stat

exec(open("ShapeGeneration.py").read())

np.random.seed(1)


# ===== parameter to be set by user =====

n_geo = 4000        # number of different geometries

n_shape_mu    = 3   # expected number of sampled shapes
n_shape_sigma = 1   # standard deviation of number of sampled shapes

L_ref       =  0.5  # reference size for sampled shapes
minTriAngle = 15    # minimum inside angle of triangle

n_res_x = 64        # x resolution of binary geometry image
n_res_y = 64        # y resolution of binary geometry image

datasetID = 2       # for name of data set, and to specify the data set during the network
                    #  training process


# ===== other parameters that one could play with =====

delta = 1e-3                                 # spacing length scale for point cloud
                                             #  defining geometry surface. used
                                             #  during the generation of the dataset
tol   = np.power(np.finfo(float).eps, 0.75)  # tolerance how far the points in the
                                             #  point cloud can be placed inside the
                                             #  actual geomtry
print_sampling = False                       # print details on sampled primitive
                                             #  primitive shapes


# ===== prepare data generation =====

x_  = np.linspace(0,1,num=n_res_x)
y_  = np.linspace(0,1,num=n_res_y)
x,y = np.meshgrid(x_,y_)

Z   = np.zeros((n_geo,n_res_y ,n_res_x))

n_shape = np.round(np.random.normal(n_shape_mu,n_shape_sigma,n_geo))
nNonPos = np.sum(n_shape<=0)
n_shape[n_shape<=0] = 1


# ===== generate training data =====

np.random.seed(3210)
t0 = time.time()
t1 = time.time()
print('  geo    >>  n_pts   n_in    >>  dt_geo   t_elaps   t_total  t_remain')
for i_geo in range(n_geo):
    z = createSampleBinClass(x,y,Z[i_geo],L_ref,n_shape[i_geo])
    Z[i_geo] = z
    t2  = time.time()
    dt1 = t2-t1
    dt0 = t2-t0
    dtT = dt0*n_geo/(i_geo+1)
    dtF = (n_geo-i_geo-1)/(i_geo+1)*dt0
    t1  = t2
    print('{:5.0f}    >>  {:5.0f}  {:5.0f}    >>  {:6.2f}  {:8.1f}  {:8.1f}  {:8.1f}'.format(
        i_geo+1,z.size,np.sum(z==1),dt1,dt0,dtT,dtF),flush=True)


# ===== save training data =====

np.save('../data/train-data/{:02.0f}_indi'.format(datasetID),Z,allow_pickle=False)


# ===== make a plot =====

i_geo = n_geo-1

fig = plt.figure()

plt.imshow(Z[i_geo],extent=[0,1,0,1],cmap="binary")
plt.colorbar()
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top  =False, labelbottom=False)
plt.tick_params(axis='y', which='both', left  =False, right=False, labelleft  =False)

plt.show()
