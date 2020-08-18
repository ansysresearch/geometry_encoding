import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from network import UNet


dir_checkpoint = 'checkpoints/'

resolution = 200
X = np.load("data/X_1obj.npy").astype(float).reshape((-1, 1, resolution, resolution))
X = torch.from_numpy(X)
Y = np.load("data/Y_1obj.npy").reshape((-1, 1, resolution, resolution))
Y = torch.from_numpy(Y)
data_ds = TensorDataset(X, Y)

val_percent = 0.1
test_percent = 0.1
batch_size = 10
lr = 0.01

n_val = int(len(data_ds) * val_percent)
n_test = int(len(data_ds) * test_percent)
n_train = len(data_ds) - n_val - n_test
train, val, test = random_split(data_ds, [n_train, n_val, n_test])

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=1, n_classes=1, bilinear=True).to(device=device)

optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=3)
loss_fn = nn.L1Loss()

num_epochs = 10
for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end ="")
    net.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32)
        pred = net(xb)
        loss = loss_fn(pred, yb)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

    print(f", training loss= {epoch_loss}", end="")

    epoch_lossv = 0
    for xbv, ybv in val_loader:
        xbv = xbv.to(device=device, dtype=torch.float32)
        ybv = ybv.to(device=device, dtype=torch.float32)
        predv = net(xbv)
        lossv = loss_fn(predv, ybv)
        epoch_lossv += lossv.item()

    scheduler.step(epoch_lossv)
    print(f", validation loss= {epoch_lossv}")

    if epoch % 5 == 4:
        os.mkdir(dir_checkpoint)
        torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
