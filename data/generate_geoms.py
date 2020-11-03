import numpy as np
from data.geoms import Circle, nGon, Rectangle, Diamond, CrossX


def generate_one_geometry(obj_list, xmax=0.8, ymax=0.8):
    obj_id = np.random.choice(obj_list)
    if obj_id == "Circle":
        r = max(0.2, np.random.random() * xmax)
        geom = Circle(r)
    elif obj_id == "Rectangle":
        w = max(0.2, np.random.random() * xmax)
        h = max(0.2, np.random.random() * ymax)
        geom = Rectangle([w, h])
    elif obj_id == "nGon":
        n_vertices = np.random.choice([5])
        vertices = []
        for i in range(n_vertices):
            th = np.random.random() * np.pi / n_vertices + 2 * i * np.pi / n_vertices
            r = max(0.4, np.random.random() * xmax)
            vertices.append([r*np.cos(th), r*np.sin(th)])
        geom = nGon(vertices)
    elif obj_id == "Diamond":
        rx = max(0.2, np.random.random() * xmax)
        ry = max(0.2, np.random.random() * ymax)
        geom = Diamond([rx, ry])
    elif obj_id == "CrossX":
        w = max(0.2, np.random.random() * xmax)
        r = max(0.2, np.random.random() * xmax*0.3)
        geom = CrossX([w, r])
    else:
        print(obj_id)
        raise("object %s is not yet implemented" % obj_id)
    return geom


def augment_geometry_1(geom, mode="all"):
    if mode == "all":
        mode = np.random.choice(["rotate", "translate", "scale"])

    if mode == "rotate":
        th = np.random.random() * np.pi * 2
        return geom.copy().rotate(th)
    elif mode == "translate":
        t1 = np.random.random() - 0.5
        t2 = np.random.random() - 0.5
        return geom.copy().translate((t1, t2))
    elif mode == "scale":
        s = np.random.random() * 1.5 + 0.5
        return geom.copy().scale(s)
    else:
        raise(ValueError("Mode %s is not recognized"%mode))


def augment_geometry_2(sdfs, n_obj=200):
    idx = np.random.randint(0, len(sdfs), n_obj)
    r = np.random.random((n_obj, 1, 1)) * 0.1 + 0.1
    sdfs_with_hole = sdfs[idx, :, :].copy()
    sdfs_with_hole = abs(sdfs_with_hole) - r
    sdfs_updated = np.concatenate([sdfs, sdfs_with_hole])
    return sdfs_updated


def augment_geometry_3(sdfs, n_obj=200):
    idx = np.random.randint(0, len(sdfs), n_obj)
    r = np.random.random((n_obj, 1, 1)) * 0.05 + 0.05
    sdfs_rounded = sdfs[idx, :, :].copy()
    sdfs_rounded = sdfs_rounded - r
    sdfs_updated = np.concatenate([sdfs, sdfs_rounded])
    return sdfs_updated


def generate_geometries(n_obj=500, n_aug=3, obj_list=("Circle", "Rectangle", "Diamond", "Cross", "nGon")):

    print("generating objects")
    geoms = [generate_one_geometry(obj_list) for _ in range(n_obj)]

    # geoms is centered at origin. we randomly translate all geoms
    geoms = [augment_geometry_1(g, mode="translate") for g in geoms]

    # now produce replicates of geoms with random rotation, scaling or translation
    geoms_aug = []
    print("augmenting with rotation, translation and scaling")
    for _ in range(n_aug):
        geoms_aug += [augment_geometry_1(g) for g in geoms]
    return geoms_aug


def combine_geometries(geoms, n1, n2, x, y):
    # n1 is number of geomtries in each combination
    # n2 is the number of combinations
    # e.g. n1=2, n3=3 which give [(g1, g2), (g3,g4), (g5, g6)]

    random_idx = np.random.randint(0, len(geoms)-1, n1*n2).reshape(n2, n1)
    combined_geoms = []
    for idx in random_idx:
        combined_geoms.append([geoms[i] for i in idx])

    combined_sdf = []
    for geom in combined_geoms:
        sdf = [g.eval_sdf(x, y) for g in geom]
        sdf = np.min(sdf, axis=0)
        combined_sdf.append(sdf)
    return combined_geoms, combined_sdf


def sample_near_geometry(geoms, n_sample, lb=-np.inf, ub=np.inf):
    sample_pnts = []
    if n_sample == 0:
        return sample_pnts
    while True:
        # generate two long random vectors,
        x = np.random.random(10000) * 2 - 1
        y = np.random.random(10000) * 2 - 1

        # compute sdf
        if isinstance(geoms, list):
            sdf = np.min([geom.eval_sdf(x, y) for geom in geoms], axis=0)
        else:
            sdf = geoms.eval_sdf(x, y)

        # find those that satisfy condition  lb < sdf < ub
        mask = np.logical_and(sdf < ub, sdf > lb)
        new_pnts = np.array([x[mask], y[mask], sdf[mask]])

        # update point matrix
        if len(sample_pnts) == 0:
            sample_pnts = new_pnts
        else:
            sample_pnts = np.concatenate([sample_pnts, new_pnts], axis=1)
        if sample_pnts.shape[1] >= n_sample:
            break

    return sample_pnts[:, :n_sample]


def filter_sdfs(sdf):
    if np.mean(sdf < 0) > 0.85: #object is too big
        return False
    elif np.mean(sdf < 0) < 0.1: #object is too small
        return False
    elif np.any(np.min(sdf < 0, axis=0)):  #object extends an entire row
        return False
    elif np.any(np.min(sdf < 0, axis=1)):  #object extends an entire column
        return False
    else:
        return True


# def generate_offgrid_data(obj_list, img_resolution=100, n_obj=50, n_points=1000, save_name=None):
#     up_factor = 4
#     img_fine_resolution = img_resolution * up_factor
#     imgs, sdfs = generate_data(obj_list, img_resolution=img_fine_resolution, n_obj=n_obj)
#     x = np.linspace(-1, 1, img_fine_resolution)
#     y = np.linspace(-1, 1, img_fine_resolution)
#
#     pnts = []
#     for sdf in sdfs:
#         idx = np.random.randint(0, img_fine_resolution-1, n_points)
#         idy = np.random.randint(0, img_fine_resolution-1, n_points)
#         xi = x[idx]
#         yi = y[idy]
#         sdfi = sdf[idx, idy]
#         pnts.append([xi, yi, sdfi])
#
#     pnts = np.array(pnts)
#     if save_name:
#         np.save(data_folder + "pnt_" + save_name + ".npy", pnts)
#
#     return pnts


# def generate_data_scipy_method(file_name):
#     imgs = np.load(data_folder + "img_" + file_name + ".npy")
#     np.save(data_folder + "img_scipy_" + file_name + ".npy", imgs)
#
#     scipy_sdfs = []
#     for img in imgs:
#         s1, s2 = img.shape
#         assert s1 == s2
#         assert np.all(np.unique(img) == [0., 1.])
#         scipy_sdf = -distance_transform_edt(img) + distance_transform_edt(1 - img)
#         scipy_sdf /= (s1 // 2)
#         scipy_sdfs.append(scipy_sdf)
#     scipy_sdfs = np.array(scipy_sdfs)
#     np.save(data_folder + "sdf_scipy_" + file_name + ".npy", scipy_sdfs)
#
#     pnts = np.load(data_folder + "pnt_" + file_name + ".npy")
#     np.save(data_folder + "pnt_scipy_" + file_name + ".npy", pnts)




