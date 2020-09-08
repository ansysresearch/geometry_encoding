import numpy as np
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from network import get_network
from utils import read_data, plot_interpolation_results
from params import *

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


test_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0, end_suffix="_test")
test_interpolate_data = interpolate_sdf(test_ds, n=500)
save_name = prediction_save_dir + "test_interpolate_data" + save_name + ".npy"
np.save(save_name, test_interpolate_data)

# train_ds, _ = read_data(dataset_id, with_pnts=True, val_frac=0)
# train_interpolate_data = interpolate_sdf(train_ds)
# save_name = prediction_save_dir + "train_interpolate_data" + save_name + ".npy"
# np.save(save_name, test_interpolate_data, train_interpolate_data)
