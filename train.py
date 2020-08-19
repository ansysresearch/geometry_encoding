import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from network import get_network
from parseArgs import args

## parsing input parameters
network_id     = args.network_id
save_name      = args.save_name
dataset_id     = args.dataset_id
num_epochs     = args.num_epochs
save_every     = args.save_every
batch_size     = args.batch_size
lr             = args.learning_rate
checkpoint_dir = args.checkpoint_dir
img_resolution = args.img_resolution
val_frac       = args.val_frac

X = np.load("data/datasets/X_" + dataset_id + ".npy").astype(float)
X = X.reshape((-1, 1, img_resolution, img_resolution))
X = torch.from_numpy(X)
Y = np.load("data/datasets/Y_" + dataset_id + ".npy")
Y = Y.reshape((-1, 1, img_resolution, img_resolution))
Y = torch.from_numpy(Y)
data_ds = TensorDataset(X, Y)

## split to train and validation sets.
n_val = int(len(data_ds) * val_frac)
n_train = len(data_ds) - n_val
train, val = random_split(data_ds, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

## Read network and setup optimizer, loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id).to(device=device) # UNet(n_channels=1, n_classes=1, bilinear=True).to(device=device)
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_fn = nn.L1Loss()

# train
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

    print(f", training loss= {epoch_loss/len(train_loader)}", end="")

    epoch_lossv = 0
    for xbv, ybv in val_loader:
        xbv = xbv.to(device=device, dtype=torch.float32)
        ybv = ybv.to(device=device, dtype=torch.float32)
        predv = net(xbv)
        lossv = loss_fn(predv, ybv)
        epoch_lossv += lossv.item()

    scheduler.step(epoch_lossv)
    print(f", validation loss= {epoch_lossv/len(val_loader)}")

    if epoch % save_every == save_every - 1:
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(net.state_dict(), checkpoint_dir + save_name + ".pth")
