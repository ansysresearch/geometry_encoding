import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from network import get_network
from utils import read_data
from params import *
import matplotlib.pyplot as plt

dtype      = DATA_TYPE
network_id = NETWORK_ID
dataset_id = DATASET_ID
save_name  = NETWORK_SAVE_NAME
n_prediction   = COMPUTE_PREDICTIONS_NUM
network_save_dir = NETWORK_SAVE_DIRECTORY
prediction_save_dir = PREDICTION_SAVE_DIRECTORY

device = 'cpu'
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
net.eval()


def interpolate_sdf(ds, n=10):
    img_res = ds[0][0].shape[-1]
    x = np.linspace(-1, 1, img_res)
    y = np.linspace(-1, 1, img_res)
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
    save_list = np.array(save_list, dtype=object)
    return save_list




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


test_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0, end_suffix="_test")
test_interpolate_data = interpolate_sdf(test_ds, n=500)
save_name = prediction_save_dir + "test_interpolate_data" + save_name + ".npy"
np.save(save_name, test_interpolate_data)

# train_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0)
# train_interpolate_data = interpolate_sdf(train_ds)
# save_name = prediction_save_dir + "train_interpolate_data" + save_name + ".npy"
# np.save(save_name, test_interpolate_data, train_interpolate_data)
