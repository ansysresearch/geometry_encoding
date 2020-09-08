import os
import torch.nn as nn
from torch import optim

from params import *
from utils import read_data, find_best_gpu
from network import get_network
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dtype = DATA_TYPE
network_id = NETWORK_ID
dataset_id = DATASET_ID
save_name  = NETWORK_SAVE_NAME
num_epochs = NUM_EPOCHS
save_every = SAVE_EVERY
batch_size = BATCH_SIZE
lr         = LEARNING_RATE
val_frac   = VALIDATION_FRACTION
scheduler_patience = LEARNING_RATE_PLATEAU_PATIENCE
scheduler_factor   = LEARNING_RATE_PLATEAU_FACTOR
scheduler_min_lr   = MINIMUM_LEARNING_RATE
checkpoint_dir     = CHECKPOINTS_DIRECTORY
network_save_dir   = NETWORK_SAVE_DIRECTORY
runs_save_dir      = RUNS_SAVE_DIRECTORY

# read data
train_ds, val_ds = read_data(dataset_id, val_frac=val_frac)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

# set cpu/gpu device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    gpu_id = find_best_gpu()
    if gpu_id:
        torch.cuda.set_device(gpu_id)

# read network and setup optimizer, loss
net = get_network(network_id).to(device=device, dtype=dtype)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_factor,
                                                 verbose=True, min_lr=scheduler_min_lr)
loss_fn = nn.L1Loss()

writer = SummaryWriter(runs_save_dir + save_name)

# train
for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end="")
    net.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=dtype)
        yb = yb.to(device=device, dtype=dtype)
        optimizer.zero_grad()
        pred = net(xb)
        loss = loss_fn(pred, yb)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(train_loader)

    print("training loss=%0.4f,  " % epoch_loss, end="")
    writer.add_scalar("Loss/train", epoch_loss, epoch)

    epoch_lossv = 0
    for xbv, ybv in val_loader:
        xbv = xbv.to(device=device, dtype=dtype)
        ybv = ybv.to(device=device, dtype=dtype)
        predv = net(xbv)
        lossv = loss_fn(predv, ybv)
        epoch_lossv += lossv.item()

    epoch_lossv /= len(val_loader)
    scheduler.step(epoch_lossv)

    print("validation loss=%0.4f,  " % epoch_lossv)
    writer.add_scalar("Loss/val", epoch_lossv, epoch)

    if epoch % save_every == save_every - 1:
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(net.state_dict(), network_save_dir + save_name + ".pth")

writer.flush()
writer.close()
