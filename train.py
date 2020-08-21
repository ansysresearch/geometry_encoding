import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from network import get_network
import argparse

# training arg parser
default_name = time.ctime().replace(" ", "_").replace(":", "")
training_parser = argparse.ArgumentParser()
training_parser.add_argument("-nw", "--network-id", type=int, help="network id",       default=1)
training_parser.add_argument("-s",  "--save-name",  type=str, help="save name",        default=default_name)
training_parser.add_argument("-d",  "--dataset-id", type=str, help="dataset id",       default="1")
training_parser.add_argument("-ne", "--num-epochs", type=int, help="number of epochs", default=100)
training_parser.add_argument("-se", "--save-every", type=int, help="save every",       default=10)
training_parser.add_argument("-b",  "--batch-size", type=int, help="batch size",       default=50)
training_parser.add_argument("-vf", "--val-frac",   type=float, help="validation fraction", default=0.1)
training_parser.add_argument("-lr", "--learning-rate", type=float, help="learning rate", default=0.01)
training_args = training_parser.parse_args()

## parsing input parameters
network_id     = training_args.network_id
save_name      = training_args.save_name
dataset_id     = training_args.dataset_id
num_epochs     = training_args.num_epochs
save_every     = training_args.save_every
batch_size     = training_args.batch_size
lr             = training_args.learning_rate
val_frac       = training_args.val_frac
checkpoint_dir = "checkpoints/"

# read data
X = np.load("data/datasets/X_" + dataset_id + ".npy").astype(float)
img_resolution = X.shape[-1]
X = X.reshape((-1, 1, img_resolution, img_resolution))
X = torch.from_numpy(X)
Y = np.load("data/datasets/Y_" + dataset_id + ".npy")
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
net = get_network(network_id).to(device=device) # UNet(n_channels=1, n_classes=1, bilinear=True).to(device=device)
#optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_fn = nn.L1Loss()

def custom_loss_fn(y_pred, y_true, mask):
    l1_loss = nn.L1Loss(reduction='none')
    loss = l1_loss(y_pred, y_true)
    return torch.mean(loss * mask)


loss_mask = torch.zeros((img_resolution, img_resolution))
loss_mask[img_resolution-2:, :] = 1
loss_mask[:2, :] = 1
loss_mask[:, img_resolution-2:] = 1
loss_mask[:, :2] = 1
loss_mask *= 5
loss_mask += 1
loss_mask = loss_mask.to(device=device)

#print(net.num_paras())
# train
for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end="")
    net.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32)
        pred = net(xb)
        #loss = loss_fn(pred, yb)
        loss = custom_loss_fn(pred, yb, loss_mask)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

    print(f"training loss= {epoch_loss/len(train_loader)},  ", end="")

    epoch_lossv = 0
    for xbv, ybv in val_loader:
        xbv = xbv.to(device=device, dtype=torch.float32)
        ybv = ybv.to(device=device, dtype=torch.float32)
        predv = net(xbv)
        lossv = custom_loss_fn(predv, ybv, loss_mask)
        #lossv = loss_fn(predv, ybv)
        epoch_lossv += lossv.item()

    scheduler.step(epoch_lossv)
    print(f"validation loss= {epoch_lossv/len(val_loader)}")

    if epoch % save_every == save_every - 1:
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(net.state_dict(), checkpoint_dir + save_name + ".pth")
