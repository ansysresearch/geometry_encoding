import os
import sys
import cv2
import numpy as np
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

        plt.text(-0.7, 3.0, "bw image", fontdict={"size": 14}, transform=plt.gca().transAxes)
        plt.text(-0.7, 1.5, "ground truth", fontdict={"size": 14}, transform=plt.gca().transAxes)
        plt.text(-0.7, 0.5, "prediction", fontdict={"size": 14}, transform=plt.gca().transAxes)
        plt.text(2, 3.5, "comparison on horizontal lines", fontdict={"size": 14}, transform=plt.gca().transAxes)
        plt.text(2, 1.6, "comparison on vertical lines", fontdict={"size": 14}, transform=plt.gca().transAxes)
        plt.text(1.8, 3.7, "dashed lines = ground truth, solid lines = prediction", fontdict={"size": 14}, transform=plt.gca().transAxes)

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


