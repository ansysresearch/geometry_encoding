import os
from data import *
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

_obj_list_3d = ("Sphere", "Ellipsoid", "Capsule", "Cylinder", "Box",
                "RoundedBox", "HollowBox", "Torus", "Octahedron")
_obj_list_2d = ("Circle", "nGon", "Rectangle", "Diamond")


def get_img_acc(img_res, level=4):
    for i in range(level):
        img_res *= 2
        img_res -= 1
    return img_res


def generate_dataset(args, test_dataset=False, obj_list=_obj_list_2d, show=True,
                     accurate=True, return_random_points=True):
    n_obj = args.n_obj // 10 if test_dataset else args.n_obj
    dataset_id = args.dataset_id + str(args.img_res)
    save_name = dataset_id + "_test" if test_dataset else dataset_id
    img_res = args.img_res
    data_folder = args.data_folder

    # grid coordiantes
    level = 4  # how many times more accurate
    img_res = get_img_acc(img_res, level=level) if accurate else img_res
    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))

    # generate all geometries, and their sdfs
    print("generating objects")
    geoms1 = generate_geometries(n_obj=n_obj, obj_list=obj_list)
    sdf1 = [g.eval_sdf(x, y) for g in geoms1]

    # create geometries that are composed of two or three objects, compute their sdf
    # print("augmenting objects")
    # geoms2, sdf2 = combine_geometries(geoms1, 2, 2*n_obj, x, y)
    # geoms3, sdf3 = combine_geometries(geoms1, 3, 3*n_obj, x, y)
    # sdfs = np.array(sdf1 + sdf2 + sdf3)
    #
    # # filter some strange looking geometries.
    # mask = [filter_sdfs(s) for s in sdfs]
    # sdfs = sdfs[mask, :, :]

    print("augmenting objects")
    img1 = (np.array(sdf1) < 0).astype(int)
    img2 = combine_imgs(img1, 2, 2*n_obj)
    img3 = combine_imgs(img1, 3, 3*n_obj)
    imgs = np.concatenate((img1, img2, img3), axis=0)

    # filter some strange looking geometries.
    mask = [filter_imgs(s) for s in imgs]
    imgs = imgs[mask, :, :]

    # note that analytical value of sdf computed in the geoms.py leads to
    # artifact in the sdf values inside geometries, when they are combined.
    # this is because boolean operation do not produce a correct sdf, but only a lowerbound for it.
    # see https://www.iquilezles.org/www/articles/interiordistance/interiordistance.htm
    # I will use the analytical values of sdf to generate black-white images, then use
    # two euclidean distance transforms to compute the signed distance function.

    numerical_imgs = []
    numerical_sdfs = []
    random_points = []
    for img in imgs:
        if return_random_points:
            if accurate:
                x = np.linspace(-1, 1, img_res)
                xidx = np.random.randint(0, img_res, 10000)
                yidx = np.random.randint(0, img_res, 10000)
                numerical_sdf = compute_numerical_sdf(img)
                random_point = np.array([x[xidx], x[yidx], numerical_sdf[yidx, xidx]]).T
            else:
                numerical_sdf, random_point = compute_numerical_sdf(img, return_random_points=True)
            random_points.append(random_point)
        else:
            numerical_sdf = compute_numerical_sdf(img)

        if accurate: numerical_sdf = numerical_sdf[::2**level, ::2**level]
        numerical_imgs.append(numerical_sdf < 0)
        numerical_sdfs.append(numerical_sdf)
    numerical_imgs = np.array(numerical_imgs)
    numerical_sdfs = np.array(numerical_sdfs)
    if return_random_points: random_points = np.array(random_points)

    if show:
        for idx in np.random.randint(0, len(numerical_sdfs), 10):
            plot_sdf(numerical_sdfs[-idx, :, :] < 0, numerical_sdfs[-idx, :, :], show=not return_random_points)
            if return_random_points:
                plt.subplot(2, 2, 3)
                plt.tricontourf(random_points[-idx, :, 0], random_points[-idx, :, 1], random_points[-idx, :, 2], 50,
                                cmap='hot')
                plt.tricontour(random_points[-idx, :, 0], random_points[-idx, :, 1], random_points[-idx, :, 2],
                               levels=30, colors='k')
                plt.show()

    np.save(os.path.join(data_folder, "img_" + save_name + ".npy"), numerical_imgs)
    np.save(os.path.join(data_folder, "sdf_" + save_name + ".npy"), numerical_sdfs)
    if return_random_points: np.save(os.path.join(data_folder, "pnts_" + save_name + ".npy"), random_points)


def generate_dataset_accurate(args, test_dataset=False, obj_list=_obj_list_2d, return_random_points=True, show=True):
    n_obj = args.n_obj // 10 if test_dataset else args.n_obj
    dataset_id = args.dataset_id + str(args.img_res)
    save_name = dataset_id + "_test" if test_dataset else dataset_id
    img_res = args.img_res
    data_folder = args.data_folder

    # grid coordiantes
    assert img_res == 128, "fix the size of high resolution. this is set only for 128 now."
    img_res_acc = 2033
    x, y = np.meshgrid(np.linspace(-1, 1, img_res_acc), np.linspace(-1, 1, img_res_acc))

    # generate all geometries, and their sdfs
    print("generating objects")
    geoms1 = generate_geometries(n_obj=n_obj, obj_list=obj_list)
    sdf1 = [g.eval_sdf(x, y) for g in geoms1]

    # create geometries that are composed of two or three objects, compute their sdf
    print("augmenting objects")
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
    # I will use the analytical values of sdf to generate black-white images, then use
    # two euclidean distance transforms to compute the signed distance function.

    imgs = []
    numerical_sdfs = []
    random_points = []
    for sdf in sdfs:
        img = sdf < 0
        numerical_sdf = compute_numerical_sdf(img)
        if return_random_points:
            x = np.linspace(-1, 1, img_res_acc)
            xidx = np.random.randint(0, img_res_acc, 10000)
            yidx = np.random.randint(0, img_res_acc, 10000)
            random_point = np.array([x[xidx], x[yidx], numerical_sdf[yidx, xidx]]).T
            random_points.append(random_point)

        numerical_sdf = numerical_sdf[::16, ::16]
        img = numerical_sdf < 0
        imgs.append(img)
        numerical_sdfs.append(numerical_sdf)

    imgs = np.array(imgs)
    numerical_sdfs = np.array(numerical_sdfs)
    if return_random_points: random_points = np.array(random_points)

    if show:
        for idx in np.random.randint(0, len(numerical_sdfs), 10):
            plot_sdf(numerical_sdfs[-idx, :, :] < 0, numerical_sdfs[-idx, :, :], show=not return_random_points)
            if return_random_points:
                plt.subplot(2, 2, 3)
                plt.tricontourf(random_points[-idx, :, 0], random_points[-idx, :, 1], random_points[-idx, :, 2], 50,
                                cmap='hot')
                plt.tricontour(random_points[-idx, :, 0], random_points[-idx, :, 1], random_points[-idx, :, 2],
                               levels=30, colors='k')
                plt.show()

    np.save(os.path.join(data_folder, "img_" + save_name + ".npy"), imgs)
    np.save(os.path.join(data_folder, "sdf_" + save_name + ".npy"), numerical_sdfs)
    if return_random_points: np.save(os.path.join(data_folder, "pnts_" + save_name + ".npy"), random_points)


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