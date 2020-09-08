import time
from tqdm import tqdm
from src.ShapeGeneration import *


# ===== prepare data generation =====

x_  = np.linspace(0,1,num=n_res_x)
y_  = np.linspace(0,1,num=n_res_y)
x,y = np.meshgrid(x_,y_)

if illu_mode:
    n_geo      =   1
    n_res_illu = 100
    n_samp     = n_res_illu*n_res_illu
    n_ep_dat   =   1
    x_illu_  = np.linspace(0,1,num=n_res_illu)
    y_illu_  = np.linspace(0,1,num=n_res_illu)
    x_illu,y_illu = np.meshgrid(x_illu_,y_illu_)
    x_illu   = np.reshape(x_illu,(n_samp,1))
    y_illu   = np.reshape(y_illu,(n_samp,1))

Z   = np.zeros((n_geo,n_res_y ,n_res_x))
SDF = np.zeros((n_geo,n_ep_dat,n_samp ,3))

n_shape = np.round(np.random.normal(n_shape_mu,n_shape_sigma,n_geo))
nNonPos = np.sum(n_shape<=0)
n_shape[n_shape<=0] = 1


# ===== generate training data =====

np.random.seed(210)
t0 = time.time()
t1 = time.time()
for i_geo in range(n_geo):
    if illu_mode:
        z, xyd = createSampleIllu(x,y,Z[i_geo],L_ref,n_shape[i_geo],n_samp,n_ep_dat,x_illu,y_illu)
    else:
        z, xyd = createSample(x,y,Z[i_geo],L_ref,n_shape[i_geo],n_samp,n_ep_dat)
    Z[i_geo] = z
    SDF[i_geo] = np.reshape(xyd,(n_ep_dat,n_samp,3))
    t2  = time.time()
    dt1 = t2-t1
    dt0 = t2-t0
    dtT = dt0*n_geo/(i_geo+1)
    dtF = (n_geo-i_geo-1)/(i_geo+1)*dt0
    t1  = t2
    print('{:5.0f}\t>>  {:5.0f}  {:5.0f}\t>>  {:5.2f}  {:8.1f}  {:8.1f}  {:8.1f}'.format(
        i_geo+1,z.size,np.sum(z==1),dt1,dt0,dtT,dtF),flush=True)


# ===== save training data =====

if not illu_mode:
    np.save('../data/jans-data/{:02.0f}_indi'.format(datasetID),Z  ,allow_pickle=False)
    np.save('../data/jans-data/{:02.0f}_sdfc'.format(datasetID),SDF,allow_pickle=False)


# ===== make a plot =====

i_geo = n_geo-1
xyd = SDF[i_geo,:,:,:]

n_ep_plt = n_ep_dat  # how many epoch of data to plot and use for sdf contour plot
x_samples = np.reshape(xyd[:n_ep_plt,:,0],(n_samp*n_ep_plt,))
y_samples = np.reshape(xyd[:n_ep_plt,:,1],(n_samp*n_ep_plt,))
s_samples = np.reshape(xyd[:n_ep_plt,:,2],(n_samp*n_ep_plt,))
if illu_mode:
    s_gridpts = np.reshape(xyd[:n_ep_plt,:,2],(n_res_illu,n_res_illu))
vMax = np.max(np.abs(s_samples))
levels = np.linspace(-vMax,vMax,num=25)

fig = plt.figure()
fig.set_size_inches(12, 4)

plt.subplot(1,2,1)
plt.imshow(Z[i_geo],extent=[0,1,0,1],cmap="binary")
plt.colorbar()
plt.gca().invert_yaxis()
plt.tick_params(axis='x', which='both', bottom=False, top  =False, labelbottom=False)
plt.tick_params(axis='y', which='both', left  =False, right=False, labelleft  =False)
if not illu_mode:
    plt.plot(x_samples, 1.0-y_samples, 'ro', ms=1)

plt.subplot(1,2,2)
if illu_mode:
    plt.imshow(s_gridpts,extent=[0,1,0,1],cmap="RdBu_r",vmin=-vMax,vmax=vMax)
    plt.colorbar()
    plt.gca().invert_yaxis()
else:
    plt.tricontour(x_samples, y_samples, s_samples, levels=levels, linewidths=0.5, colors='k')
    cntr2 = plt.tricontourf(x_samples, y_samples, s_samples, levels=levels, cmap="RdBu_r")
    plt.colorbar(cntr2)
    ax = plt.gca()
    ax.set_aspect(1.0)
plt.tick_params(axis='x', which='both', bottom=False, top  =False, labelbottom=False)
plt.tick_params(axis='y', which='both', left  =False, right=False, labelleft  =False)

plt.show()
