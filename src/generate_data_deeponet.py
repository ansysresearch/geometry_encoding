import torch
import pickle
from data import *
from src import parse_arguments
from src.network import get_network
from scipy.interpolate import griddata, RectBivariateSpline


def prepare_data_for_nonlinear_deeponet_interpolator(args, n_fine=8, idx_start=0):
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    network_id = args.net_id
    save_name = args.net_id + '_' + args.dataset_id + str(args.img_res)
    checkpoint_dir = args.ckpt_dir
    network_save_dir = checkpoint_dir + 'networks/'
    device = 'cpu'
    net = get_network(network_id=network_id).to(device=device, dtype=dtype)
    net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
    net.eval()

    img_res = args.img_res
    img_res_fine = img_res * n_fine
    data_folder = args.data_folder
    save_name_fine = args.dataset_id + str(img_res_fine)

    x_linear = np.linspace(-1, 1, img_res)
    x, y = np.meshgrid(x_linear, x_linear)
    grid_flatten = np.stack((x.flatten(), y.flatten()), axis=-1)

    x_fine_linear = np.linspace(-1, 1, img_res_fine)
    x_fine, y_fine = np.meshgrid(x_fine_linear, x_fine_linear)
    fine_grid_flatten = np.stack((x_fine.flatten(), y_fine.flatten()),  axis=-1)

    imgs_fine = np.load(data_folder + "img_" + save_name_fine + ".npy")
    sdfs_fine = np.load(data_folder + "sdf_" + save_name_fine + ".npy")

    for i, (img_fine, sdf_fine) in enumerate(zip(imgs_fine, sdfs_fine)):
        interpolator = RectBivariateSpline(x_fine_linear, x_fine_linear, img_fine)
        img = np.array([interpolator(yi, xi).item() for (xi, yi) in grid_flatten])
        img = np.round(img)
        # img = griddata(fine_grid_flatten, img_fine.flatten(), grid_flatten)
        img = img.reshape(img_res, img_res)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        with torch.no_grad():
            sdf_pred_tensor = net(img_tensor)
        sdf_pred = sdf_pred_tensor.squeeze().numpy()

        random_row_idx = np.random.randint(0, img_res_fine, 10000).tolist()
        random_col_idx = np.random.randint(0, img_res_fine, 10000).tolist()
        random_x = x_fine_linear[random_row_idx]
        random_y = x_fine_linear[random_col_idx]
        random_img = img_fine[(random_col_idx, random_row_idx)].astype(int)
        random_sdf = sdf_fine[(random_col_idx, random_row_idx)]
        random_matrix = np.stack((random_x, random_y, random_img, random_sdf), axis=-1)

        data_dict = {"img": img, "sdf_pred": sdf_pred, "random_points": random_matrix}
        pickle_file_name = r'C:/Users/amaleki/Downloads/sdf-data/deeponet_data/data_dict_%d.pickle' % (i + idx_start +1)
        with open(pickle_file_name, 'wb') as fid:
            pickle.dump(data_dict, fid, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_data_for_nonlinear_deeponet_interpolator(args, n_fine=8, idx_start=727)

