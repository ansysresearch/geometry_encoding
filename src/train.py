import os
from src import (read_data, TrainLogger, get_dtype, get_device, get_loss_func, get_save_name, get_optimizer)
from src.network import get_network

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(args):
    # read args

    dtype              = get_dtype(args)
    network_id         = args.net_id
    save_name          = get_save_name(args)
    num_epochs         = args.n_epochs
    save_every         = args.save_every
    batch_size         = args.batch_size
    loss_fn            = args.loss_fn
    checkpoint_dir     = args.ckpt_dir
    network_save_dir   = os.path.join(checkpoint_dir, 'networks')
    runs_save_dir      = os.path.join(checkpoint_dir, 'runs')

    # read data
    train_ds, val_ds = read_data(args)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, pin_memory=True, drop_last=True)

    # set cpu/gpu device
    device = get_device(args)

    # read network and setup optimizer, loss
    loss_fn = get_loss_func(loss_fn)
    net = get_network(network_id).to(device=device, dtype=dtype)
    optimizer, scheduler = get_optimizer(net, args)

    tf_writer = SummaryWriter(os.path.join(runs_save_dir, save_name))
    train_log_writer = TrainLogger(os.path.join(runs_save_dir, save_name + "_training_logs"), optimizer)

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device=device, dtype=dtype), yb.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)
        tf_writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_lossv = 0
        net.eval()
        for xbv, ybv in val_loader:
            xbv, ybv = xbv.to(device=device, dtype=dtype), ybv.to(device=device, dtype=dtype)
            predv = net(xbv)
            lossv = loss_fn(predv, ybv)
            epoch_lossv += lossv.item()

        epoch_lossv /= len(val_loader)
        scheduler.step(epoch_lossv)

        tf_writer.add_scalar("Loss/val", epoch_lossv, epoch)

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            print(f"epoch {epoch},  ", end="")
            print("training loss=%0.4f,  " % epoch_loss, end="")
            print("validation loss=%0.4f,  " % epoch_lossv)
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(net.state_dict(), os.path.join(network_save_dir, save_name + ".pth"))
            train_log_writer.write_training_step(epoch)
    tf_writer.flush()
    tf_writer.close()
    train_log_writer.close()


