import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset


def read_data(dataset_id, val_frac=0.1, end_suffix="", with_pnts=False):

    img_file_name = "data/datasets/img_" + dataset_id + end_suffix + ".npy"
    sdf_file_name = "data/datasets/sdf_" + dataset_id + end_suffix + ".npy"

    # read image (regular grid) data
    X_img = np.load(img_file_name).astype(float)
    img_resolution = X_img.shape[-1]
    X_img = X_img.reshape((-1, 1, img_resolution, img_resolution))
    X_img = torch.from_numpy(X_img)
    Y_img = np.load(sdf_file_name)
    Y_img = Y_img.reshape((-1, 1, img_resolution, img_resolution))
    Y_img = torch.from_numpy(Y_img)

    # split to train and validation sets.
    n_train = int(X_img.shape[0] * (1 - val_frac))
    perm_idx = np.random.permutation(X_img.shape[0])
    train_idx = perm_idx[:n_train]
    val_idx = perm_idx[n_train:]
    X_img_train, X_img_val = X_img[train_idx, :, :], X_img[val_idx, :, :]
    Y_img_train, Y_img_val = Y_img[train_idx, :, :], Y_img[val_idx, :, :]

    if with_pnts:
        # read pnt (irregular points) data
        pnts_file_name = "data/datasets/pnt_" + dataset_id + end_suffix + ".npy"
        pnts = np.load(pnts_file_name)[:, :, 400:]
        X_pnts = pnts[:, :2, :]
        X_pnts = torch.from_numpy(X_pnts)
        Y_pnts = pnts[:, 2, :]
        Y_pnts = torch.from_numpy(Y_pnts)
        X_pnts_train, X_pnts_val = X_pnts[train_idx, :, :], X_pnts[val_idx, :, :]
        Y_pnts_train, Y_pnts_val = Y_pnts[train_idx, :], Y_pnts[val_idx, :]
        train_ds = TensorDataset(X_img_train, Y_img_train, X_pnts_train, Y_pnts_train)
        val_ds = TensorDataset(X_img_val, Y_img_val, X_pnts_val, Y_pnts_val)
    else:
        train_ds = TensorDataset(X_img_train, Y_img_train)
        val_ds = TensorDataset(X_img_val, Y_img_val)

    # set data loaders

    return train_ds, val_ds


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        return gpu_id


def plot_data(file_name):
    test_preds = np.load(file_name)
    img_resolution = test_preds[0].shape[-1]
    xx = np.linspace(0, img_resolution-1, img_resolution)
    sampling_lines = [img_resolution // 10, img_resolution // 2, int(img_resolution * 0.9)]
    clr_list = ['r', 'b', 'g']
    for sdf, sdf_pred in test_preds:
        plt.figure(figsize=(12, 10))
        img = sdf < 0
        plt.subplot(3, 2, 1)
        plt.imshow(img, cmap='binary')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        levels = np.linspace(sdf.min() + 0.02, sdf.max() - 0.2, 5)
        plt.subplot(3, 2, 3)
        plt.imshow(sdf, cmap='hot')
        #plt.colorbar()
        cn = plt.contour(sdf, levels, colors='k')
        plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)  # contour line labels
        plt.contour(sdf, [0], colors='k', linewidths=3)
        #plt.clabel(cn0, fmt='%0.2f', colors='k')  # contour line labels
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, [xx[plt_idx]] * img_resolution, clr + '--')
            plt.plot([xx[plt_idx]] * img_resolution, xx, clr + '--')

        plt.subplot(3, 2, 5)
        plt.imshow(sdf_pred, cmap='hot')
        #plt.colorbar()
        cn = plt.contour(sdf_pred, levels, colors='k')
        plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)  # contour line labels
        plt.contour(sdf_pred, [0], colors='k', linewidths=3)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,2,2)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, sdf[plt_idx, :], clr + '--')
            plt.plot(xx, sdf_pred[plt_idx, :], clr + '-')

        plt.subplot(2,2,4)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(sdf[:, plt_idx], xx, clr + '--')
            plt.plot(sdf_pred[:, plt_idx], xx, clr + '-')

        plt.show()