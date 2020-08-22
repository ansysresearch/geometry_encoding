import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from network import get_network
import argparse

network_id = "UNet"
save_name  = "circ50"
dataset_id = "circ50"
num_epochs = 100
save_every = 20
batch_size = 50
lr         = 0.001
val_frac   = 0.2
scheduler_patience = 3
scheduler_min_lr = 5e-6
checkpoint_dir = "checkpoints/"

# read data
X = np.load("data/datasets/img_" + dataset_id + ".npy").astype(float)
img_resolution = X.shape[-1]
X = X.reshape((-1, 1, img_resolution, img_resolution))
X = torch.from_numpy(X)
Y = np.load("data/datasets/sdf_" + dataset_id + ".npy")
Y = Y.reshape((-1, 1, img_resolution, img_resolution))
Y = torch.from_numpy(Y)
data_ds = TensorDataset(X, Y)

# split to train and validation sets.
n_val = int(len(data_ds) * val_frac)
n_train = len(data_ds) - n_val
train, val = random_split(data_ds, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# read network and setup optimizer, loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id).to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True, min_lr=scheduler_min_lr)


# custom loss function
def custom_loss_fn(y_pred, y_true):
    l1_val = nn.L1Loss()(y_pred, y_true)
    out_pred = (y_pred > 0).type(torch.float64)
    #out_pred.requires_grad = True
    out = (y_pred > 0).type(torch.float64)
    ce_val = nn.BCELoss()(out_pred, out)
    return l1_val, ce_val


loss_fn = custom_loss_fn
writer = SummaryWriter()
gamma = 1e4
# train
for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end="")
    net.train()
    epoch_loss, epoch_l1_loss, epoch_bce_loss = 0, 0, 0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        pred = net(xb)
        l1_loss, bce_loss = loss_fn(pred, yb)
        loss = l1_loss + gamma * bce_loss
        epoch_l1_loss += l1_loss.item()
        epoch_bce_loss += bce_loss.item()
        epoch_loss += loss.item()
        loss.backward()
        #nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
    epoch_l1_loss  /= len(train_loader)
    epoch_bce_loss /= len(train_loader)
    epoch_loss     /= len(train_loader)

    print("training: l1_loss=%0.4f, bce_loss=%0.4f, loss=%0.4f,  " % (epoch_l1_loss, epoch_bce_loss, epoch_loss), end="")
    writer.add_scalar("Loss/train", epoch_loss, epoch)

    epoch_lossv, epoch_l1_lossv, epoch_bce_lossv = 0, 0, 0
    for xbv, ybv in val_loader:
        xbv = xbv.to(device=device, dtype=torch.float32)
        ybv = ybv.to(device=device, dtype=torch.float32)
        predv = net(xbv)

        l1_lossv, bce_lossv = loss_fn(predv, ybv)
        lossv = l1_lossv + gamma * bce_lossv
        epoch_l1_lossv += l1_lossv.item()
        epoch_bce_lossv += bce_lossv.item()
        epoch_lossv += lossv.item()

    scheduler.step(epoch_lossv)
    epoch_l1_lossv /= len(val_loader)
    epoch_bce_lossv /= len(val_loader)
    epoch_lossv /= len(val_loader)

    print("validation: l1_loss=%0.4f, bce_loss=%0.4f, loss=%0.4f,  " % (epoch_l1_lossv, epoch_bce_lossv, epoch_lossv))
    writer.add_scalar("Loss/valid", epoch_lossv, epoch)

    if epoch % save_every == save_every - 1:
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(net.state_dict(), checkpoint_dir + save_name + ".pth")

writer.flush()
writer.close()