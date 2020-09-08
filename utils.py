import os
import sys
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from params import *

data_folder = DATA_FOLDER


def read_data(dataset_id, val_frac=0.1, end_suffix="", with_pnts=False):
    img_file_name = data_folder + "img_" + dataset_id + end_suffix + ".npy"
    sdf_file_name = data_folder + "sdf_" + dataset_id + end_suffix + ".npy"

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
        pnts_file_name = data_folder + "pnt_" + dataset_id + end_suffix + ".npy"
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

    return train_ds, val_ds


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = np.argmax(memory_available).item()
        return gpu_id


def plot_prediction_results(file_name, show=True, colorbar=False):
    test_preds = np.load(file_name)
    img_resolution = test_preds[0].shape[-1]
    xx = np.linspace(0, img_resolution-1, img_resolution)
    sampling_lines = [int(img_resolution * 0.1), int(img_resolution * 0.5), int(img_resolution * 0.9)]
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
        if colorbar: plt.colorbar()
        cn = plt.contour(sdf, levels, colors='k')
        plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)  # contour line labels
        plt.contour(sdf, [0], colors='k', linewidths=3)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, [xx[plt_idx]] * img_resolution, clr + '--')
            plt.plot([xx[plt_idx]] * img_resolution, xx, clr + '--')

        plt.subplot(3, 2, 5)
        plt.imshow(sdf_pred, cmap='hot')
        if colorbar: plt.colorbar()
        cn = plt.contour(sdf_pred, levels, colors='k')
        plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)  # contour line labels
        plt.contour(sdf_pred, [0], colors='k', linewidths=3)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 2, 2)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, sdf[plt_idx, :], clr + '--')
            plt.plot(xx, sdf_pred[plt_idx, :], clr + '-')

        plt.subplot(2, 2, 4)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(sdf[:, plt_idx], xx, clr + '--')
            plt.plot(sdf_pred[:, plt_idx], xx, clr + '-')

        if show:
            plt.show()


def aggregate_interpolation_results(data):
    errors = []
    errors_rel = []
    for img, xp, yp, sdf, sdf_pred in data:
        error = sdf - sdf_pred
        error_rel = (sdf - sdf_pred) / (abs(sdf) + 0.01)
        errors.append(error)
        errors_rel.append(error_rel)
    return np.array(errors), np.array(errors_rel)


def plot_interpolation_results(file_name, n=10):
    data = np.load(file_name, allow_pickle=True)

    er, er_rel = aggregate_interpolation_results(data)
    er_m = er.mean(axis=1)
    plt.plot(np.arange(len(er_m)), er_m, 'b.--')
    plt.gca().fill_between(np.arange(len(er_m)), er_m - er.std(axis=1), er_m + er.std(axis=1), color='b', alpha=.1)
    plt.xticks([], [])
    plt.title("average error for 500 geometries \n computed over 600 randomly sampled points")
    plt.xlabel("geometries")
    plt.ylabel("sdf - predicted sdf")
    plt.ylim(-0.03, 0.03)
    plt.show()

    er_rel_m = er_rel.mean(axis=1)
    plt.plot(np.arange(len(er_rel_m)), er_rel_m, 'b.--')
    plt.gca().fill_between(np.arange(len(er_rel_m)), er_rel_m - er_rel.std(axis=1), er_rel_m + er_rel.std(axis=1),
                           color='b', alpha=.1)
    plt.xticks([], [])
    plt.title("average relative error for 500 geometries \n computed over 600 randomly sampled points")
    plt.xlabel("geometries")
    plt.ylabel("(sdf - predicted sdf)/ (abs(sdf) + 0.01)")
    plt.ylim(-0.1, 0.1)

    for idx in np.random.randint(0, data.shape[0], n):
        img, xp, yp, sdf, sdf_pred = data[idx]
        img = img.squeeze()

        err = sdf - sdf_pred
        err_rel = np.minimum(1, err / (abs(sdf) + 0.01))
        err_log = np.log10(abs(err))
        err_rel_log = np.log10(abs(err_rel))
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap="binary")
        plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(-1, 1, 5))
        plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(-1, 1, 5))
        plt.gca().invert_yaxis()
        plt.scatter((xp + 1) * img.shape[0] / 2, (yp + 1) * img.shape[1] / 2, c=err, s=10, norm=LogNorm())
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.hist(err_log, bins=np.linspace(-10, 0, 20), weights=np.ones_like(sdf)/sdf.shape[0])
        plt.xlabel("log(error)")
        plt.ylabel("frequency ratio")
        plt.show()