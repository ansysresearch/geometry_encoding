import os
import sys
import cv2
import torch
import pickle
import datetime
import numpy as np
from src.network import get_network
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
    # this function finds the GPU with most free memory.
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
    To compute labels for encoder network, we may need to run unet model on the entire dataset.
    whole data does not fit on GPU, we should run it in batches
    """
    s1, s2, s3, s4 = X.shape
    assert s2 == 1 and s3 == s4, "check batch_run_unet_model function"
    if s1 < batch_size:
        X2 = X.to(device=device, dtype=dtype)
        Y2 = model(X2)
    else:
        n_extra = s1 % batch_size
        if n_extra != 0:
            print("throughing out the last %d number of data" % n_extra)
            X = X[:-n_extra, ...]
        X2 = X.reshape(-1, batch_size, s2, s3, s4)
        Y2 = []
        for xx in X2:
            xx = xx.to(device=device, dtype=dtype)
            yy = model(xx)
            # yy = yy.unsqueeze(0)
            Y2.append(yy)
        Y2 = torch.cat(Y2, dim=0)
    return Y2.to(device='cpu')


def prepare_training_data(X, Y, args):
    """
    this function prepares training data
    autoencoder value is
      0: this is for training the unet or unet + autoencoder, inputs are binary img, outputs are true sdf
      1: this is for training the autoencoder alone, inputs and outputs are true sdf
      2: this is for training autoencoder alone, inputs and outputs are ouputs of unet
      3: this is for training autoencoder alone, inputs are unet outputs, outputs are true sdf
      4: this is for training deeponet alone, similar to 1
      5: this is for training deeponet alone, similar to 2

    :param X: img data
    :param Y: true sdf data
    """
    model_flag = args.model_flag
    data_network_id = args.data_network_id
    if model_flag == 0:
        return X, Y
    elif model_flag in [1, 4]:
        return Y, Y
    else:
        dtype = get_dtype(args)
        device = get_device(args, get_best_cuda=False)

        # get unet model
        checkpoint_dir = args.ckpt_dir
        network_save_dir = os.path.join(checkpoint_dir, 'networks')
        data_network_save_name = [d for d in os.listdir(network_save_dir) if data_network_id in d]
        if len(data_network_save_name) == 0:
            raise(RuntimeError("no model with name %s exists" % data_network_id))
        elif len(data_network_save_name) > 1:
            print("multiple %s found" % data_network_id)
        data_network_model_address = os.path.join(network_save_dir, data_network_save_name[0])
        data_network_model = get_network(network_id=data_network_id.split("_")[0]).to(device=device, dtype=dtype)
        data_network_model.load_state_dict(torch.load(data_network_model_address, map_location=device))
        data_network_model.eval()

        # can't load all data on GPU, must divide into batches
        with torch.no_grad():
            Y_tmp = batch_run_model(data_network_model, X, device, dtype)
            if model_flag in [2, 5]:
                return Y_tmp, Y_tmp
            elif model_flag == 3:
                return Y_tmp, Y
            else:
                raise(ValueError("model_flag value %d is not recognized" %model_flag))


def read_data(args, end_suffix="", with_random_points=False):
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

    if with_random_points:
        rnd_pnts_file_name = os.path.join(data_folder, "pnts_" + dataset_id + end_suffix + ".npy")
        rnd_pnts = np.load(rnd_pnts_file_name).astype(float)
        Xp = torch.from_numpy(rnd_pnts[:, :, :-1])
        Yp = torch.from_numpy(rnd_pnts[:, :, -1])
        Xp_train, Xp_val = Xp[train_idx, ...], Xp[val_idx, ...]
        Yp_train, Yp_val = Yp[train_idx, ...], Yp[val_idx, ...]
        train_ds = TensorDataset(X_train, Xp_train, Yp_train)
        val_ds = TensorDataset(X_val, Xp_val, Yp_val)
    else:
        train_ds = TensorDataset(X_train, Y_train)
        val_ds = TensorDataset(X_val, Y_val)

    return train_ds, val_ds


def read_data_deeponet(args, n_data=100):
    # get data for deeponet training.
    # for the args.model_flag:
    #      if equal 4: training containts % of data with all random points,
    #                  validation contains rest of data with all random points,
    #      if equal 5: training contains all data with % of random points
    #                  validation contains all data with rest of random points

    imgs        = []
    sdfs        = []
    rnd_pnts    = []
    img_res     = args.img_res
    val_frac    = args.val_frac
    data_folder = args.data_folder
    assert args.model_flag >= 4, "not a deeponet model flag."

    # for i in range(n_data):
    #     file_name = os.path.join(data_folder, "data_dict_%d.pickle" % (i+1))
    #     with open(file_name, 'rb') as fid:
    #         data_dict = pickle.load(fid)
    #         imgs.append(data_dict['img'])
    #         sdfs.append(data_dict['sdf_pred'])
    #         rnd_pnts.append(data_dict['random_points'])
    for i in range(n_data):
        file_name = os.path.join(data_folder, "data_dict_%d.pickle" % (i+1))
        with open(file_name, 'rb') as fid:
            data_dict = pickle.load(fid)
            imgs.append(data_dict['img'])
            sdfs.append(data_dict['sdf_pred'])
            rnd_pnts.append(data_dict['random_points'])


    # read image (regular grid) data
    imgs = np.array(imgs).astype(float)
    imgs = imgs.reshape((-1, 1, img_res, img_res))
    imgs = torch.from_numpy(imgs)

    sdfs = np.array(sdfs)
    sdfs = sdfs.reshape((-1, 1, img_res, img_res))
    sdfs = torch.from_numpy(sdfs)

    X, _ = prepare_training_data(imgs, sdfs, args)

    rnd_pnts = np.array(rnd_pnts)
    Xp = torch.from_numpy(rnd_pnts[:, :, :3])
    Yp = torch.from_numpy(rnd_pnts[:, :, 3])

    # split to train and validation sets.
    if args.model_flag == 4:
        n_train = int(n_data * (1 - val_frac))
        perm_idx = np.random.permutation(n_data)
        train_idx = perm_idx[:n_train]
        val_idx = perm_idx[n_train:]
        X_train, X_val = X[train_idx, ...], X[val_idx, ...]
        Xp_train, Xp_val = Xp[train_idx, ...], Xp[val_idx, ...]
        Yp_train, Yp_val = Yp[train_idx, ...], Yp[val_idx, ...]
    elif args.model_flag == 5:
        n_train = int(10000 * (1 - val_frac))
        perm_idx = np.random.permutation(n_data)
        train_idx = perm_idx[:n_train]
        val_idx = perm_idx[n_train:]
        X_train, X_val = X, X
        Xp_train, Xp_val = Xp[:, train_idx, :], Xp[:, val_idx, :]
        Yp_train, Yp_val = Yp[:, train_idx, :], Yp[:, val_idx, :]
    else:
        raise(ValueError("method %d not recognized, see deeponet data gen." %args.model_flag))

    train_ds = TensorDataset(X_train, Xp_train, Yp_train)
    val_ds = TensorDataset(X_val, Xp_val, Yp_val)

    return train_ds, val_ds


def compute_edges_img(img):
    edge_lower_tresh = 50
    edge_upper_tresh = 200
    running_img = img.copy()
    running_img *= 255
    running_img = running_img.astype(np.uint8)
    edges = cv2.Canny(running_img, edge_lower_tresh, edge_upper_tresh)
    return edges


def compute_perimeter_img(img):
    edges = compute_edges_img(img)
    perimeter = np.sum(edges) / (img.shape[0] * 2 * 255)
    return perimeter


# from scipy.spatial.distance import cdist
# def compute_sharpest_angle_img(img):
#     edges = compute_edges_img(img)
#     edges_pixels = np.array(np.where(edges == 255))
#     dist_mat  = cdist(edges_pixels, edges_pixels)
#     closest_pixels_idx = np.argmin(cdist)


