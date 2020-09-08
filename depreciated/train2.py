import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from network import get_network, get_network2, NetworkB
import matplotlib.pyplot as plt

network_id = "UNet"
dataset_id = "all128"
save_name  = network_id + "_" + dataset_id
num_epochs = 60
save_every = 20
batch_size = 150
lr         = 0.001
val_frac   = 0.2
scheduler_patience = 3
scheduler_min_lr = 5e-6
scheduler_factor = 0.2
checkpoint_dir = "checkpoints/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = NetworkB().to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)

# read img data
img_res = 32
x = np.linspace(-1, 1, img_res)
y = np.linspace(-1, 1, img_res)
X, Y = np.meshgrid(x, y)
from data.geoms import Circle
circle = Circle(0.5).translate((0.2, 0.3))
eps = 0.1
sdf = circle.eval_sdf(X, Y)
sdf_noisy = sdf + np.random.random((img_res, img_res)) * eps
loss_fn = nn.L1Loss()

for epoch in range(100):
    loss_epoch = 0
    pred_mat = np.zeros((img_res, img_res))
    for i in range(img_res):
        yi = torch.from_numpy(-y[i:i+1]).to(device=device, dtype=torch.float32)
        yi.requires_grad = True
        for j in range(img_res):
            xj = torch.from_numpy(x[j:j + 1]).to(device=device, dtype=torch.float32)
            xj.requires_grad = True
            sdfij = torch.from_numpy(np.array([sdf_noisy[i, j]])).to(device=device, dtype=torch.float32)
            data = torch.cat([xj, yi, sdfij])#.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = net(data)
            dx = torch.from_numpy(np.array([1.])).to(device=device, dtype=torch.float32)
            dy = torch.from_numpy(np.array([1.])).to(device=device, dtype=torch.float32)
            gradx = torch.autograd.grad(pred, [xj], grad_outputs=dx, create_graph=True)[0]
            grady = torch.autograd.grad(pred, [yi], grad_outputs=dy, create_graph=True)[0]
            loss = (pred - sdfij) ** 2 + 0.25 * abs(gradx ** 2 + grady ** 2 - 1.)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            pred_mat[i, j] = pred.item()
    print(loss_epoch/(128**2), end = ", ")
    print(np.linalg.norm(pred_mat - sdf), end = ", ")
    print(np.linalg.norm(pred_mat - sdf_noisy))




train_ds = TensorDataset(x[train_idx, :], y[train_idx, :], s[train_idx, :])
val_ds = TensorDataset(x[val_idx, :], y[val_idx, :], s[val_idx, :])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# read network and setup optimizer, loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = NetworkB().to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience,
                                                 factor=scheduler_factor,
                                                 verbose=True, min_lr=scheduler_min_lr)
loss_fn = nn.L1Loss()

for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end="")
    net.train()
    epoch_loss = 0
    for t_idx in train_loader:
        for jdx in range(600):
            xb, yb, sb = t_idx[0][0][jdx], t_idx[0][1][jdx], t_idx[0][2][jdx]
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            sb = sb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = net(xb, yb, sb)
            loss = loss_fn(pred, sb)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
    epoch_loss /= len(train_loader)

    print("training loss=%0.4f,  " % epoch_loss, end="")
