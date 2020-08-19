import torch
import numpy as np
from network import UNet
from data.sympy_helper import plot_sdf

resolution = 200
net = UNet(n_channels=1, n_classes=1, bilinear=True)
net.load_state_dict(torch.load("checkpoints/CP_epoch100.pth", map_location=torch.device('cpu')))
net.eval()
X = np.load("data/datasets/X_1obj.npy").astype(float).reshape((-1, 1, resolution, resolution))
X = torch.from_numpy(X)
Y = np.load("data/datasets/Y_1obj.npy").reshape((-1, 1, resolution, resolution))
Y = torch.from_numpy(Y)

idx = 5
img = X[idx, :, :, :]
img = img.unsqueeze(0)
sdf = Y[idx, :, :, :]
sdf = sdf.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = img.to(device=device, dtype=torch.float32)

with torch.no_grad():
    sdf_pred = net(img)

img = img.squeeze()
sdf = sdf.squeeze()
sdf_pred = sdf_pred.squeeze()
print(img.shape, sdf.shape, sdf_pred.shape)

import matplotlib.pyplot as plt

xx = np.linspace(0, resolution, resolution)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(sdf, cmap='hot')
plt.contour(sdf, 10, colors='k')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
for plt_idx, clr in zip([30, 100, 170], ['r', 'b', 'g']):
    plt.plot(xx, [xx[plt_idx]]*resolution, clr + '--')
    plt.plot([xx[plt_idx]]*resolution, xx, clr + '--')

# plt.show()
plt.subplot(2, 2, 2)
plt.imshow(sdf_pred, cmap='hot')
plt.contour(sdf_pred, 10, colors='k')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
for plt_idx, clr in zip([30, 100, 170], ['r', 'b', 'g']):
    plt.plot(xx, sdf[plt_idx, :], clr + '--')
    plt.plot(xx, sdf_pred[plt_idx, :], clr + '-')

plt.subplot(2,2,4)
for plt_idx, clr in zip([30, 100, 170], ['r', 'b', 'g']):
    plt.plot(sdf[:, plt_idx], xx, clr + '--')
    plt.plot(sdf_pred[:, plt_idx], xx, clr + '-')


plt.show()
