import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from network import get_network
from utils import read_data

network_id = "UNet"
dataset_id = "all50"
network_file = network_id + "_" + dataset_id

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device)
net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
net.eval()


def interpolate_sdf(ds):
    img_res = ds[0][0].shape[-1]
    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))
    for idx in np.random.randint(0, len(ds), 10):
        img, _, pnts, pnts_sdf = ds[idx]
        img = torch.unsqueeze(img, 0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            img_sdf = net(img)
        img_sdf = torch.squeeze(img_sdf).numpy()
        f_sdf = interp2d(x, y, img_sdf)

        xp = pnts[0, :]
        yp = pnts[1, :]
        sdfp_pred = np.array([f_sdf(xpi, ypi).item() for (xpi, ypi) in zip(xp, yp)])
        errorp = abs(pnts_sdf - sdfp_pred)
        plt.scatter(xp, yp, c=errorp, s=10, )
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.hist(np.log(abs(errorp)), bins=np.linspace(-10, 0, 20))
        plt.show()


train_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0)
interpolate_sdf(train_ds)

test_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0, end_suffix="_test")
interpolate_sdf(train_ds)
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
