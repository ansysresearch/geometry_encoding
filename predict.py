import torch
import numpy as np
from network import UNet
from data.sympy_helper import plot_sdf

resolution = 200
net = UNet(n_channels=1, n_classes=1, bilinear=True)
net.load_state_dict(torch.load("checkpoints/CP_epoch5.pth"))
net.eval()
X = np.load("data/X_1obj.npy").astype(float).reshape((-1, 1, resolution, resolution))
X = torch.from_numpy(X)
Y = np.load("data/Y_1obj.npy").reshape((-1, 1, resolution, resolution))
Y = torch.from_numpy(Y)

idx = 0
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

plot_sdf(img, sdf)
plot_sdf(img, sdf_pred)
