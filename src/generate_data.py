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


_obj_list_3d = ("Sphere", "Ellipsoid", "Capsule", "Cylinder", "Box",
                "RoundedBox", "HollowBox", "Torus", "Octahedron")
def generate_dataset_3d(args, test_dataset=False, obj_list=_obj_list_3d):
    n_obj = args.n_obj // 10 if test_dataset else args.n_obj
    dataset_id = args.dataset_id + str(args.img_res)
    save_name = dataset_id + "_test" if test_dataset else dataset_id
    img_res = args.img_res
    data_folder = args.data_folder

    # grid coordiantes
    x, y, z = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))

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