import os
from src import read_data, find_best_gpu, TrainLogger, get_loss_func, get_save_name
from src.network import get_network

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(args):
    # read args
    autoencoder        = args.autoencoder
    dtype              = torch.float32 if args.dtype == "float32" else torch.float64
    network_id         = args.net_id
    save_name          = get_save_name(args)
    num_epochs         = args.n_epochs
    save_every         = args.save_every
    batch_size         = args.batch_size
    loss_fn            = args.loss_fn
    lr                 = args.lr
    scheduler_patience = args.lr_patience
    scheduler_step     = args.lr_step
    scheduler_factor   = args.lr_factor
    scheduler_min_lr   = args.lr_min
    checkpoint_dir     = args.ckpt_dir
    use_cpu            = args.use_cpu
    network_save_dir   = os.path.join(checkpoint_dir, 'networks')
    runs_save_dir      = os.path.join(checkpoint_dir, 'runs')

    # read data
    train_ds, val_ds = read_data(args)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, pin_memory=True, drop_last=True)

    # set cpu/gpu device
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    if device == torch.device('cuda'):
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)

    # read network and setup optimizer, loss
    loss_fn = get_loss_func(loss_fn)
    net = get_network(network_id).to(device=device, dtype=dtype)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if scheduler_step is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=scheduler_patience,
                                                         factor=scheduler_factor,
                                                         verbose=True, min_lr=scheduler_min_lr)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_factor)
    # writers

    tf_writer = SummaryWriter(os.path.join(runs_save_dir, save_name))
    train_log_writer = TrainLogger(os.path.join(runs_save_dir, save_name + "_training_logs"), optimizer)

    # train
    for epoch in range(num_epochs):
        print(f"epoch {epoch},  ", end="")
        net.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            yb = yb.to(device=device, dtype=dtype)
            xb = yb if autoencoder else xb.to(device=device, dtype=dtype)
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
        net.eval()
        for xbv, ybv in val_loader:
            ybv = ybv.to(device=device, dtype=dtype)
            xbv = ybv if autoencoder else xbv.to(device=device, dtype=dtype)
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
            torch.save(net.state_dict(), os.path.join(network_save_dir, save_name + ".pth"))
            train_log_writer.write_training_step(epoch)
    tf_writer.flush()
    tf_writer.close()
    train_log_writer.close()


