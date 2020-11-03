import os
import torch
import torch.nn as nn
from torch import optim
from src import read_data, find_best_gpu, TrainLogger
from src.network import get_network
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(args):
    # read args
    dtype              = torch.float32 if args.dtype == "float32" else torch.float64
    network_id         = args.net_id
    dataset_id         = args.dataset_id + str(args.img_res)
    save_name          = network_id + '_' + dataset_id
    num_epochs         = args.n_epochs
    save_every         = args.save_every
    batch_size         = args.batch_size
    lr                 = args.lr
    scheduler_patience = args.lr_patience
    scheduler_factor   = args.lr_factor
    scheduler_min_lr   = args.lr_min
    checkpoint_dir     = args.ckpt_dir
    network_save_dir   = checkpoint_dir + 'networks/'
    runs_save_dir      = checkpoint_dir + 'runs/'

    # read data
    train_ds, val_ds = read_data(args)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, pin_memory=True, drop_last=True)

    # set cpu/gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)

    # read network and setup optimizer, loss
    loss_fn = nn.L1Loss()
    net = get_network(network_id).to(device=device, dtype=dtype)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience,
                                                     factor=scheduler_factor,
                                                     verbose=True, min_lr=scheduler_min_lr)
    # writers
    tf_writer = SummaryWriter(runs_save_dir + save_name)
    train_log_writer = TrainLogger("training_logs", optimizer)

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
        tf_writer.add_scalar("Loss/train", epoch_loss, epoch)

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
        tf_writer.add_scalar("Loss/val", epoch_lossv, epoch)

        if epoch % save_every == save_every - 1:
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(net.state_dict(), network_save_dir + save_name + ".pth")
            train_log_writer.write_training_step(epoch)
    tf_writer.flush()
    tf_writer.close()
    train_log_writer.close()

