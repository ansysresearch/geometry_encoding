import torch
import numpy as np
from network import get_network
import matplotlib.pyplot as plt


def plot_data(test_preds):
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

network_id = "UNet3"
network_file = "UNet3_all150"
data_name = "C:/Users/amaleki/Desktop/exotic_shapes/exotic_shapes.npy"


device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device)
net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
net.eval()
X = np.load(data_name).astype(float)
img_resolution = X.shape[-1]
X = X.reshape((-1, 1, img_resolution, img_resolution))
Y = X.copy()
X = (X < 0).astype(float)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

saved_list = []
for img, sdf in zip(X, Y):
    img = img.unsqueeze(0)
    sdf = sdf.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        sdf_pred = net(img)

    img = img.squeeze()
    sdf = sdf.squeeze()
    sdf_pred = sdf_pred.squeeze()

    saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))


save_list = np.array(saved_list)
plot_data(save_list)