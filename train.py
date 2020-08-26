import torch.nn as nn
from torch import optim
from utils import *
from network import get_network
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

network_id = "UNet"
dataset_id = "all50"
save_name  = network_id + "_" + dataset_id
num_epochs = 100
save_every = 20
batch_size = 50
lr         = 0.001
val_frac   = 0.2
scheduler_patience = 3
scheduler_min_lr = 5e-6
scheduler_factor = 0.2
checkpoint_dir = "checkpoints/"

# read data
train_ds, val_ds = read_data(dataset_id, val_frac=val_frac)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)


# set cpu/gpu device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    gpu_id = find_best_gpu()
    if gpu_id: torch.cuda.set_device(gpu_id)

print("running simulation on " + device.type)
# read network and setup optimizer, loss
net = get_network(network_id).to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience,
                                                 factor=scheduler_factor,
                                                 verbose=True, min_lr=scheduler_min_lr)
loss_fn = nn.L1Loss()
writer = SummaryWriter("runs/" + save_name)

# train
for epoch in range(num_epochs):
    print(f"epoch {epoch},  ", end="")
    net.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device, dtype=torch.float32)
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
        xbv = xbv.to(device=device, dtype=torch.float32)
        ybv = ybv.to(device=device, dtype=torch.float32)
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
        torch.save(net.state_dict(), checkpoint_dir + save_name + ".pth")

writer.flush()
writer.close()
