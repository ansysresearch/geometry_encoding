import os
import sys
import cv2
import torch
import pickle
import warnings
import datetime
import numpy as np
from src import get_network
from torch.utils.data import TensorDataset


class TrainLogger:
    def __init__(self, file_name, optim, **header_kwargs):
        fid = open(file_name, 'w')
        self.fid = fid
        self.optim = optim
        self.write_header(header_kwargs)

    def close(self):
        self.fid.close()

    def write_header(self, header_kwargs):
        self.fid.write("Training Log:\n")
        self.fid.write("optimizer: %s\n" % self.optim.__class__.__name__)
        for k, v in header_kwargs.items():
            self.fid.write("%s k: \n" % k + str(v))
        self.write_horizonta_line()

    def write_horizonta_line(self):
        self.fid.write("=" * 50)
        self.fid.write("\n")

    def write_training_step(self, itr, **kwargs):
        opt_params = self.optim.param_groups[0]
        self.fid.write("epoch %d:\n" % itr)
        for k, v in opt_params.items():
            if k != 'params':
                self.fid.write("     %s k: \n" % k + str(v))

        for k, v in kwargs.items():
            self.fid.write("     %s k: \n" % k + str(v))

        self.write_horizonta_line()


def get_optimizer(model, args):
    lr                 = args.lr
    scheduler_patience = args.lr_patience
    scheduler_step     = args.lr_step
    scheduler_factor   = args.lr_factor
    scheduler_min_lr   = args.lr_min
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler_step is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=scheduler_patience,
                                                               factor=scheduler_factor,
                                                               verbose=True,
                                                               min_lr=scheduler_min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=scheduler_step,
                                                    gamma=scheduler_factor)
    return optimizer, scheduler


def get_loss_func(loss_str):
    if loss_str == 'l1':
        loss_func = torch.nn.L1Loss()
    elif loss_str == 'l2':
        loss_func = torch.nn.MSELoss()
    else:
        raise(ValueError("loss function %s is not recognized" % loss_str))
    return loss_func


def find_best_gpu():
    r""" this function finds the GPU with most free memory."""
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        print("best gpu, %d, %f" %(gpu_id, memory_available[gpu_id]))
        return gpu_id


def get_device(args, get_best_cuda=True):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    if device == torch.device('cuda') and get_best_cuda:
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)
    return device


def get_dtype(args):
    return torch.float32 if args.dtype == "float32" else torch.float64


def get_save_name(args):
    now = datetime.datetime.now()
    dataset_id = args.dataset_id + str(args.img_res)
    network_id = args.net_id
    save_name_tag = args.save_name
    if len(save_name_tag) == 0:
        save_name_tag = '_'.join(map(str, [now.month, now.day, now.hour, now.minute]))
    save_name = network_id + '_' + dataset_id + '_' + save_name_tag
    return save_name


def batch_run_model(model, X, device, dtype, batch_size=25):
    """"
    To compute labels for compressor (autoencoder) network,
    we may need to run processor (UNet)model on the entire dataset.
    Since the whole data does not fit on GPU, we should run it in batches.
    """
    s1, s2, s3, s4 = X.shape
    assert s2 == 1 and s3 == s4, "check batch_run_unet_model function"
    if s1 < batch_size:
        X2 = X.to(device=device, dtype=dtype)
        Y2 = model(X2)
    else:
        n_extra = s1 % batch_size
        if n_extra != 0:
            warnings.warn("throughing out the last %d number of data" % n_extra)
            X = X[:-n_extra, ...]
        X2 = X.reshape(-1, batch_size, s2, s3, s4)
        Y2 = []
        for xx in X2:
            xx = xx.to(device=device, dtype=dtype)
            yy = model(xx)
            Y2.append(yy)
        Y2 = torch.cat(Y2, dim=0)
    return Y2.to(device='cpu')


def prepare_training_data(X, Y, args):
    """
    this function prepares training data.
    if args.model_flag
      processor: inputs are binary img, outputs are true sdf (for training a processor)
      compressor: inputs and outputs are ouputs of processor. The processor network is determined by
                  another argument parameter --data-network-id.
    Args
        X (torch.Tensor): input data data
        Y (torch.Tensor): ground truth data
    """
    model_flag = args.model_flag
    data_network_id = args.data_network_id
    if model_flag == "processor":
        return X, Y
    else:
        dtype = get_dtype(args)
        device = get_device(args, get_best_cuda=False)

        # get processor (UNet) model
        checkpoint_dir = args.ckpt_dir
        network_save_dir = os.path.join(checkpoint_dir, 'networks')
        data_network_save_name = [d for d in os.listdir(network_save_dir) if data_network_id in d]
        if len(data_network_save_name) == 0:
            raise(RuntimeError("no model with name %s exists" % data_network_id))
        elif len(data_network_save_name) > 1:
            warnings.warn("multiple %s found" % data_network_id)
        data_network_model_address = os.path.join(network_save_dir, data_network_save_name[0])
        data_network_model = get_network(network_id=data_network_id.split("_")[0]).to(device=device, dtype=dtype)
        data_network_model.load_state_dict(torch.load(data_network_model_address, map_location=device))
        data_network_model.eval()

        # we should run the processor to generate data
        # but can't load all data on GPU, must divide into batches
        with torch.no_grad():
            Y_processor = batch_run_model(data_network_model, X, device, dtype)
            return Y_processor, Y_processor


def read_data(args, end_suffix=""):
    val_frac = args.val_frac
    dataset_id = args.dataset_id + str(args.img_res)
    data_folder = args.data_folder
    img_res = args.img_res
    img_file_name = os.path.join(data_folder, "img_" + dataset_id + end_suffix + ".npy")
    sdf_file_name = os.path.join(data_folder, "sdf_" + dataset_id + end_suffix + ".npy")

    # read image (regular grid) data
    imgs = np.load(img_file_name).astype(float)
    imgs = imgs.reshape((-1, 1, img_res, img_res))
    imgs = torch.from_numpy(imgs)
    sdfs = np.load(sdf_file_name)
    sdfs = sdfs.reshape((-1, 1, img_res, img_res))
    sdfs = torch.from_numpy(sdfs)

    X, Y = prepare_training_data(imgs, sdfs, args)

    # split to train and validation sets.
    n_train = int(X.shape[0] * (1 - val_frac))
    perm_idx = np.random.permutation(X.shape[0])
    train_idx = perm_idx[:n_train]
    val_idx = perm_idx[n_train:]
    X_train, X_val = X[train_idx, ...], X[val_idx, ...]
    Y_train, Y_val = Y[train_idx, ...], Y[val_idx, ...]

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    return train_ds, val_ds