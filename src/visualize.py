import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_save_name


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


def viz(args):
    save_name = get_save_name(args)
    prediction_save_dir = os.path.join(args.ckpt_dir, 'predictions')

    # showing true/predicted sdf on the training set (data network has seen).
    train_prediction_file_name = os.path.join(prediction_save_dir, "train_predictions_" + save_name + ".npy")
    plot_prediction_results(train_prediction_file_name)

    # showing true/predicted sdf on the test set (data network has not seen, but similar to training set).
    test_prediction_file_name = os.path.join(prediction_save_dir, "test_predictions_" + save_name + ".npy")
    plot_prediction_results(test_prediction_file_name)

    # showing true/predicted sdf on the exotic shape set (which are not seen nor similar to the seen data).
    exotic_prediction_file_name = os.path.join(prediction_save_dir, "exotic_predictions_" + save_name + ".npy")
    if os.path.isfile(exotic_prediction_file_name):
        plot_prediction_results(exotic_prediction_file_name)