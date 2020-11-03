import torch
from data import *
from scipy.ndimage import distance_transform_edt


def generate_dataset(args, test_dataset=False, obj_list=("Circle", "nGon", "Rectangle", "Diamond")):
    n_obj = args.n_obj // 10 if test_dataset else args.n_obj
    dataset_id = args.dataset_id + str(args.img_res)
    save_name = dataset_id + "_test" if test_dataset else dataset_id
    img_res = args.img_res
    data_folder = args.data_folder

    # grid coordiantes
    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))

    # generate all geometries, and their sdfs
    geoms1 = generate_geometries(n_obj=n_obj, obj_list=obj_list)
    sdf1 = [g.eval_sdf(x, y) for g in geoms1]

    # create geometries that are composed of two or three objects, compute their sdf
    geoms2, sdf2 = combine_geometries(geoms1, 2, 2*n_obj, x, y)
    geoms3, sdf3 = combine_geometries(geoms1, 3, 3*n_obj, x, y)
    sdfs = np.array(sdf1 + sdf2 + sdf3)

    # filter some strange looking geometries.
    mask = [filter_sdfs(s) for s in sdfs]
    sdfs = sdfs[mask, :, :]

    # create geometries with holes in them.
    sdfs = augment_geometry_2(sdfs, n_obj=int(0.5*n_obj))

    # create rounded geometries.
    sdfs = augment_geometry_3(sdfs, n_obj=int(0.5*n_obj))

    # note that analytical value of sdf computed in the geoms.py leads to
    # artifact in the sdf values inside geometries, when they are combined.
    # this is because boolean operation do not produce a correct sdf, but only a lowerbound for it.
    # see https://www.iquilezles.org/www/articles/interiordistance/interiordistance.htm
    #
    # I will use the analytical values of sdf to generate black-white images, then use
    # two euclidean distance transforms to compute the signed distance function.

    imgs = []
    scipy_sdfs = []
    for sdf in sdfs:
        img = sdf < 0
        imgs.append(img)
        scipy_sdf = -distance_transform_edt(img) + distance_transform_edt(1 - img)
        scipy_sdf /= (sdfs.shape[1] // 2)
        scipy_sdfs.append(scipy_sdf)

    imgs = np.array(imgs)
    scipy_sdfs = np.array(scipy_sdfs)

    # uncomment if you want to see some of the generated img,sdf pairs
    # for idx in np.random.randint(0, n_obj*8, 10):
    #     plot_sdf(scipy_sdfs[-idx, :, :] < 0, scipy_sdfs[-idx, :, :])

    if save_name:
        np.save(data_folder + "img_" + save_name + ".npy", imgs)
        np.save(data_folder + "sdf_" + save_name + ".npy", scipy_sdfs)

    return imgs, scipy_sdfs


import pickle
from src.network import get_network
from scipy.interpolate import griddata
def prepare_data_for_nonlinear_deeponet_interpolator(args):
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
    img_res_fine = img_res * 8
    data_folder = args.data_folder
    save_name_fine = args.dataset_id + str(img_res_fine)

    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))
    grid_flatten = np.stack((x.flatten(), y.flatten()))
    x_fine, y_fine = np.meshgrid(np.linspace(-1, 1, img_res_fine), np.linspace(-1, 1, img_res_fine))
    fine_grid_flatten = np.stack((x_fine.flatten(), y_fine.flatten()))
    imgs_fine = np.load(data_folder + "img_" + save_name_fine + ".npy")
    sdfs_fine = np.load(data_folder + "sdf_" + save_name_fine + ".npy")

    fine_grid_lin = np.linspace(-1, 1, img_res_fine)

    for i, (img_fine, sdf_fine) in enumerate(zip(imgs_fine, sdfs_fine)):
        img = griddata(fine_grid_flatten, img_fine.flatten(), grid_flatten)
        img = img.reshape(img_res, img_res)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device=device, dtype=dtype)
        with torch.no_grad():
            sdf_pred_tensor = net(img_tensor)
        sdf_pred = sdf_pred_tensor.numpy()

        random_row_idx = np.random.randint(0, img_res_fine, 10000).tolist()
        random_col_idx = np.random.randint(0, img_res_fine, 10000).tolist()
        random_x = fine_grid_lin[random_row_idx]
        random_y = fine_grid_lin[random_col_idx]
        random_img = img_fine[[random_row_idx, random_col_idx]]
        random_sdf = sdf_fine[[random_row_idx, random_col_idx]]
        random_matrix = np.stack((random_x, random_y, random_img, random_sdf))

        data_dict = {"img": img, "sdf_pred": sdf_pred, "random_points": random_matrix}
        with open(r'C:/Users/amaleki/Downloads/sdf/data_dict_%d.pickle % (i+1)', 'wb') as fid:
            pickle.dump(data_dict, fid, protocol=pickle.HIGHEST_PROTOCOL)


