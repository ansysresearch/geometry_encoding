import torch
import numpy as np
from network import get_network
import matplotlib.pyplot as plt
import argparse

prediction_parser = argparse.ArgumentParser()
prediction_parser.add_argument("-n",     "--network-id",   type=int, help="network id")
prediction_parser.add_argument("-nfile", "--network-file", type=str, help="save name")
prediction_parser.add_argument("-d",     "--data-name",    type=str, help="data name")
prediction_args = prediction_parser.parse_args()

network_id = prediction_args.network_id
network_file = prediction_args.network_file
data_name = prediction_args.data_name

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device)
net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
net.eval()
X = np.load("data/datasets/X_" + data_name + ".npy").astype(float)
img_resolution = X.shape[-1]
X = X.reshape((-1, 1, img_resolution, img_resolution))
X = torch.from_numpy(X)
Y = np.load("data/datasets/Y_" + data_name + ".npy").reshape((-1, 1, img_resolution, img_resolution))
Y = torch.from_numpy(Y)

xx = np.linspace(0, img_resolution, img_resolution)
sampling_lines = [img_resolution//10, img_resolution//2, int(img_resolution*0.9)]
clr_list = ['r', 'b', 'g']
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

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(sdf, cmap='hot')
    plt.contour(sdf, 10, colors='k')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    for plt_idx, clr in zip(sampling_lines, clr_list):
        plt.plot(xx, [xx[plt_idx]]*img_resolution, clr + '--')
        plt.plot([xx[plt_idx]]*img_resolution, xx, clr + '--')

    plt.subplot(2, 2, 2)
    plt.imshow(sdf_pred, cmap='hot')
    plt.contour(sdf_pred, 10, colors='k')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 3)
    for plt_idx, clr in zip(sampling_lines, clr_list):
        plt.plot(xx, sdf[plt_idx, :], clr + '--')
        plt.plot(xx, sdf_pred[plt_idx, :], clr + '-')

    plt.subplot(2, 2, 4)
    for plt_idx, clr in zip(sampling_lines, clr_list):
        plt.plot(sdf[:, plt_idx], xx, clr + '--')
        plt.plot(sdf_pred[:, plt_idx], xx, clr + '-')

    plt.show()
