import os
import sys
import cv2
import torch
import numpy as np
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


def read_data(args, end_suffix=""):
    val_frac = args.val_frac
    dataset_id = args.dataset_id + str(args.img_res)
    data_folder = args.data_folder
    img_res = args.img_res
    img_file_name = data_folder + "img_" + dataset_id + end_suffix + ".npy"
    sdf_file_name = data_folder + "sdf_" + dataset_id + end_suffix + ".npy"

    # read image (regular grid) data
    X_img = np.load(img_file_name).astype(float)

    X_img = X_img.reshape((-1, 1, img_res, img_res))
    X_img = torch.from_numpy(X_img)
    Y_img = np.load(sdf_file_name)
    Y_img = Y_img.reshape((-1, 1, img_res, img_res))
    Y_img = torch.from_numpy(Y_img)

    # split to train and validation sets.
    n_train = int(X_img.shape[0] * (1 - val_frac))
    perm_idx = np.random.permutation(X_img.shape[0])
    train_idx = perm_idx[:n_train]
    val_idx = perm_idx[n_train:]
    X_img_train, X_img_val = X_img[train_idx, :, :], X_img[val_idx, :, :]
    Y_img_train, Y_img_val = Y_img[train_idx, :, :], Y_img[val_idx, :, :]

    train_ds = TensorDataset(X_img_train, Y_img_train)
    val_ds = TensorDataset(X_img_val, Y_img_val)

    return train_ds, val_ds


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        print("best gpu, %d, %f" %(gpu_id, memory_available[gpu_id]))
        return gpu_id


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


