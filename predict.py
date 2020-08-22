import torch
import numpy as np
from network import get_network
import matplotlib.pyplot as plt
import argparse


def plot_data():
    test_preds = np.load("checkpoints/test_predictions.npy")
    img_resolution = test_preds[0].shape[-1]
    xx = np.linspace(0, img_resolution, img_resolution)
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

        plt.subplot(3, 2, 3)
        plt.imshow(sdf, cmap='hot')
        plt.colorbar()
        plt.contour(sdf, 10, colors='k')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, [xx[plt_idx]] * img_resolution, clr + '--')
            plt.plot([xx[plt_idx]] * img_resolution, xx, clr + '--')

        plt.subplot(3, 2, 4)
        plt.imshow(sdf_pred, cmap='hot')
        plt.colorbar()
        plt.contour(sdf_pred, 10, colors='k')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 2, 5)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(xx, sdf[plt_idx, :], clr + '--')
            plt.plot(xx, sdf_pred[plt_idx, :], clr + '-')

        plt.subplot(3, 2, 6)
        for plt_idx, clr in zip(sampling_lines, clr_list):
            plt.plot(sdf[:, plt_idx], xx, clr + '--')
            plt.plot(sdf_pred[:, plt_idx], xx, clr + '-')

        plt.show()


network_id = "UNet"
network_file = "circ50"
data_name = "circ50"
plot_arg = 1

if plot_arg == 2:
    plot_data()
else:
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_network(network_id=network_id).to(device=device)
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
    net.eval()
    X = np.load("data/datasets/img_" + data_name + ".npy").astype(float)
    img_resolution = X.shape[-1]
    X = X.reshape((-1, 1, img_resolution, img_resolution))
    X = torch.from_numpy(X)
    Y = np.load("data/datasets/sdf_" + data_name + ".npy").reshape((-1, 1, img_resolution, img_resolution))
    Y = torch.from_numpy(Y)

    saved_list = []
    for idx in np.random.randint(0, X.shape[0], 10):
        img = X[idx, :, :, :]
        img = img.unsqueeze(0)
        sdf = Y[idx, :, :, :]
        sdf = sdf.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            sdf_pred = net(img)

        img = img.squeeze()
        sdf = sdf.squeeze()
        sdf_pred = sdf_pred.squeeze()

        saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))

    np.save("checkpoints/test_predictions.npy", np.array(saved_list))
    if plot_arg == 1:
        plot_data()

    X_test = np.load("data/datasets/img_" + data_name + "-test.npy").astype(float)
    X_test = X_test.reshape((-1, 1, img_resolution, img_resolution))
    X_test = torch.from_numpy(X_test)
    Y_test = np.load("data/datasets/sdf_" + data_name + "-test.npy").reshape((-1, 1, img_resolution, img_resolution))
    Y_test = torch.from_numpy(Y_test)

    saved_list = []
    for idx in np.random.randint(0, X_test.shape[0], 10):
        img = X_test[idx, :, :, :]
        img = img.unsqueeze(0)
        sdf = Y_test[idx, :, :, :]
        sdf = sdf.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            sdf_pred = net(img)

        img = img.squeeze()
        sdf = sdf.squeeze()
        sdf_pred = sdf_pred.squeeze()

        saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))

    np.save("checkpoints/test_predictions.npy", np.array(saved_list))
    if plot_arg == 1:
        plot_data()