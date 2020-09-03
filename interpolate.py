import torch
import numpy as np
from tqdm import tqdm
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from network import get_network
from utils import read_data

dtype = torch.float32
network_id = "UNet6"
dataset_id = "all128"
network_file = "UNet6_all128"

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
try:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
except RuntimeError:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth", map_location=device))
net.eval()


def interpolate_sdf(ds, n=10):
    img_res = ds[0][0].shape[-1]
    x = np.linspace(-1, 1, img_res)
    y = np.linspace(-1, 1, img_res)
    # x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))
    save_list = []
    itr = 0
    for idx in tqdm(np.random.randint(0, len(ds), n)):
        img, _, pnts, pnts_sdf = ds[idx]
        img = torch.unsqueeze(img, 0)
        img = img.to(device=device, dtype=dtype)
        with torch.no_grad():
            img_sdf = net(img)
        img_sdf = torch.squeeze(img_sdf).numpy()
        f_sdf = RectBivariateSpline(x, y, img_sdf)

        xp = pnts[0, :]
        yp = pnts[1, :]
        pnts_sdfp_pred = np.array([f_sdf(ypi, xpi).item() for (xpi, ypi) in zip(xp, yp)])

        save_list.append((img.numpy(), xp.numpy(), yp.numpy(), pnts_sdf.numpy(), pnts_sdfp_pred))
        itr += 1
    return np.array(save_list, dtype=object)


def plot_interpolation_results(file_name, n=10):
    data = np.load(file_name, allow_pickle=True)

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
        plt.scatter((xp + 1) * img.shape[0] / 2, (yp + 1) * img.shape[1] / 2, c=err, s=10, norm=matplotlib.colors.LogNorm())
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.hist(err_log, bins=np.linspace(-10, 0, 20), weights=np.ones_like(sdf)/sdf.shape[0])
        plt.xlabel("log(error)")
        plt.ylabel("frequency ratio")
        plt.show()


def aggregate_results(file_name):
    data = np.load(file_name, allow_pickle=True)
    errors = []
    errors_rel = []
    for img, xp, yp, sdf, sdf_pred in data:
        error = sdf - sdf_pred
        error_rel = (sdf - sdf_pred) / (abs(sdf) + 0.01)
        errors.append(error)
        errors_rel.append(error_rel)
    return np.array(errors), np.array(errors_rel)

# train_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0)
# train_interpolate_data = interpolate_sdf(train_ds)
# np.save("train_interpolate_data.npy", train_interpolate_data)
# plot_interpolation_results("train_interpolate_data.npy")

test_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0, end_suffix="_test")
test_interpolate_data = interpolate_sdf(test_ds, n=500)


save_name = "checkpoints/test_interpolate_data" + network_file + ".npy"
np.save(save_name, test_interpolate_data)
# plot_interpolation_results(save_name)

er, er_rel = aggregate_results(save_name)
er_m = er.mean(axis=1)
plt.plot(np.arange(len(er_m)), er_m, 'b.--')
plt.gca().fill_between(np.arange(len(er_m)), er_m - er.std(axis=1), er_m + er.std(axis=1), color='b', alpha=.1)
plt.xticks([], [])# np.arange(len(er_m)), np.arange(len(er_m)))
plt.title("average error for 500 geometries \n computed over 600 randomly sampled points")
plt.xlabel("geometries")
plt.ylabel("sdf - predicted sdf")
plt.ylim(-0.03, 0.03)
plt.show()

er_rel_m = er_rel.mean(axis=1)
plt.plot(np.arange(len(er_rel_m)), er_rel_m, 'b.--')
plt.gca().fill_between(np.arange(len(er_rel_m)), er_rel_m - er_rel.std(axis=1), er_rel_m + er_rel.std(axis=1), color='b', alpha=.1)
plt.xticks([], [])
plt.title("average relative error for 500 geometries \n computed over 600 randomly sampled points")
plt.xlabel("geometries")
plt.ylabel("(sdf - predicted sdf)/ (abs(sdf) + 0.01)")
plt.ylim(-0.1, 0.1)
plt.show()

plot_interpolation_results(save_name)

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
