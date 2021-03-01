import os
from data import *
_obj_list_2d = ("Circle", "nGon", "Rectangle", "Diamond")


def generate_dataset(args, test_dataset=False, show=False):
    """
    generate dataset from primitve shapes.
    use sympy to compute analytical value of SDF, with random parameters.
    2D object list contains: `Circle`, `nGon`, `Rectangle`, `Diamond`
    note that analytical value of sdf leads to artifact in the sdf values inside geometries, when they are combined.
    this is because boolean operation do not produce a correct sdf, but only a lowerbound for it.
    see https://www.iquilezles.org/www/articles/interiordistance/interiordistance.htm
    we use the analytical values of sdf to generate black-white images, then use
    two euclidean distance transforms to compute the signed distance function.

    Args
        args: input arguments
        test_dataset (bool, optional), if :obj:`True`, a test dataset is generation, default :obj:`False`
        show (bool, optional), if :obj:`True`, data will be plotted,
    """
    obj_list = _obj_list_2d
    n_obj = args.n_obj // 10 if test_dataset else args.n_obj
    dataset_id = args.dataset_id
    save_name = dataset_id + "_test" if test_dataset else dataset_id
    img_res = args.img_res
    data_folder = args.data_folder

    # grid coordiantes
    x, y = np.meshgrid(np.linspace(-1, 1, img_res), np.linspace(-1, 1, img_res))

    # generate all geometries, and their sdfs
    print("generating objects")
    geoms1 = generate_geometries(n_obj=n_obj, obj_list=obj_list)
    sdf1 = [g.eval_sdf(x, y) for g in geoms1]

    # create geometries that are composed of two or three objects, compute their sdf
    print("augmenting objects")
    img1 = (np.array(sdf1) < 0).astype(int)
    img2 = combine_imgs(img1, 2, 2*n_obj)
    img3 = combine_imgs(img1, 3, 3*n_obj)
    imgs = np.concatenate((img1, img2, img3), axis=0)

    # filter some strange looking geometries.
    mask = [filter_imgs(s) for s in imgs]
    imgs = imgs[mask, :, :]

    # compute the numerical value of sdf
    numerical_imgs = []
    numerical_sdfs = []
    for img in imgs:
        numerical_sdf = compute_numerical_sdf(img)
        numerical_imgs.append(numerical_sdf < 0)
        numerical_sdfs.append(numerical_sdf)
    numerical_imgs = np.array(numerical_imgs)
    numerical_sdfs = np.array(numerical_sdfs)

    if show:
        for idx in np.random.randint(0, len(numerical_sdfs), 10):
            plot_sdf(numerical_sdfs[-idx, :, :] < 0, numerical_sdfs[-idx, :, :], show=True)

    np.save(os.path.join(data_folder, "img_" + save_name + ".npy"), numerical_imgs)
    np.save(os.path.join(data_folder, "sdf_" + save_name + ".npy"), numerical_sdfs)