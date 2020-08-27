import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
from network import get_network
from utils import read_data

network_id = "UNet"
dataset_id = "all128"
network_file = network_id + "_" + dataset_id

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device)
try:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
except RuntimeError:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth", map_location=device))
net.eval()


def interpolate_sdf(ds, n=10):
    img_res = ds[0][0].shape[-1]
    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))
    save_list = []
    itr = 0
    for idx in tqdm(np.random.randint(0, len(ds), n)):
        img, _, pnts, pnts_sdf = ds[idx]
        img = torch.unsqueeze(img, 0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            img_sdf = net(img)
        img_sdf = torch.squeeze(img_sdf).numpy()
        f_sdf = interp2d(x, y, img_sdf)

        xp = pnts[0, :]
        yp = pnts[1, :]
        pnts_sdfp_pred = np.array([f_sdf(xpi, ypi).item() for (xpi, ypi) in zip(xp, yp)])

        save_list.append((img, xp, yp, pnts_sdf, pnts_sdfp_pred))
        itr += 1
    return np.array(save_list, dtype=object)


def plot_interpolation_results(file_name, n=10):
    data = np.load(file_name, allow_pickle=True)

    for idx in np.random.randint(0, data.shape[0], n):
        img, xp, yp, sdf, sdf_pred = data[idx]
        img = img.squeeze()

        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap="binary")
        plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(-1, 1, 5))
        plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(-1, 1, 5))
        plt.gca().invert_yaxis()
        plt.scatter((xp + 1) * img.shape[0] / 2, (yp + 1) * img.shape[1] / 2, c=sdf - sdf_pred, s=10)
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.hist(np.log(abs(sdf - sdf_pred)), bins=np.linspace(-10, 0, 20), weights=np.ones_like(sdf)/sdf.shape[0])
        plt.xlabel("log(error)")
        plt.ylabel("frequency ratio")
        plt.show()


# train_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0)

# train_interpolate_data = interpolate_sdf(train_ds)
# np.save("train_interpolate_data.npy", train_interpolate_data)
# plot_interpolation_results("train_interpolate_data.npy")

test_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0, end_suffix="_test")
test_interpolate_data = interpolate_sdf(test_ds, n=50)
np.save("checkpoints/test_interpolate_data.npy", test_interpolate_data)
plot_interpolation_results("checkpoints/test_interpolate_data.npy")

#
# for idx in np.random.randint(0, 200, 10):
#     X_idx = X[idx, :, :, :]
#     X_idx = torch.from_numpy(X_idx)
#     X_idx = torch.unsqueeze(X_idx, 0)
#     X_idx = X_idx.to(device=device, dtype=torch.float32)
#     with torch.no_grad():
#         Y_idx = net(X_idx)
#     Y_idx = torch.squeeze(Y_idx).numpy()
#     f_sdf = interp2d(x, y, Y_idx)
#
#     pnt = pnts[idx, :, :]
#     xp = pnt[0, :]
#     yp = pnt[1, :]
#     zp = pnt[2, :]
#     zp_pred = np.array([f_sdf(xpi, ypi).item() for (xpi, ypi) in zip(xp, yp)])
#
#     errorp = zp - zp_pred
#     plt.scatter(xp, yp, c=errorp, s=10, )
#     plt.colorbar()
#     plt.show()
#     plt.figure()
#     plt.hist(np.log(abs(errorp)), bins=np.linspace(-10, 0, 20))
#     plt.show()
