import copy
import numpy as np

from data.geoms import Circle, nGon, Rectangle, Diamond, CrossX


def generate_one_geometry(obj_list, xmax=0.8, ymax=0.8):
    """
    randomly generate a geometry from the list `obj_list`.
    Args:
        obj_list (list of str): a list of possible objects.
        xmax, ymax (float): paramter to control maximum possible size of object.
    return: an instance of a :class:geoms.Geom
    """
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
        r = max(0.2, np.random.random() * xmax * 0.3)
        geom = CrossX([w, r])
    else:
        print(obj_id)
        raise("object %s is not yet implemented" % str(obj_id))
    return geom


def augment_geometry(geom, mode="all2"):
    """
    randomly modify an instance of :class:geoms.Geom
    returns modified object
    """
    if mode == "all2":
        mode = np.random.choice(["rotate", "translate", "scale"])
    elif mode == "all3":
        mode = np.random.choice(["rotate", "translate", "scale", "elongate", "onion", "roundify"])

    if mode == "rotate":
        th = np.random.random() * np.pi * 2
        if hasattr(geom, 'z'):
            plane = np.random.choice(['xy', 'xz', 'yz'])
            return geom.copy().rotate(th, plane=plane)
        else:
            return geom.copy().rotate(th)
    elif mode == "translate":
        t1 = np.random.random() - 0.5
        t2 = np.random.random() - 0.5
        if hasattr(geom, 'z'):
            t3 = np.random.random() - 0.5
            return geom.copy().tranlate((t1, t2, t3))
        else:
            return geom.copy().translate((t1, t2))
    elif mode == "scale":
        s = np.random.random() * 1.5 + 0.5
        return geom.copy().scale(s)
    elif mode == "elongate":
        elong_fact = np.random.random() * 0.8 + 1.1
        elong_axis = np.random.choice(["x", "y", "z"])
        return geom.copy().elongate(elong_fact, along=elong_axis)
    elif mode == "roundify":
        r = np.random.random() * 0.2 + 0.1
        return geom.copy().roundify(r)
    elif mode == "onion":
        th = np.random.random() * 0.4 + 0.1
        return geom.copy().onion(th)
    else:
        raise(ValueError("Mode %s is not recognized"%mode))


def generate_geometries(n_obj=500, n_aug=3, obj_list=("Circle", "Rectangle", "Diamond", "nGon")):
    """
    generate a random list of `n_obj` objects from the list `obj_list`.
    then added `n_aug` many augmentations.
    """
    geoms = [generate_one_geometry(obj_list) for _ in range(n_obj)]

    # geoms is centered at origin. we randomly translate all geoms
    geoms = [augment_geometry(g, mode="translate") for g in geoms]
    geoms_all = copy.deepcopy(geoms)

    # now produce replicates of geoms with random rotation, scaling or translation
    for _ in range(n_aug):
        geoms_all += [augment_geometry(g) for g in geoms]
    return geoms_all


def combine_imgs(imgs, n1, n2):
    """
    combine objects of list geoms
    Args
        imgs (3d nd.arrays): a list of images
        n1 (int): number of geometries in each combination
        n2 (int): number of combinations

    e.g. if imgs = [g1, g2, g3, g4], n1=2, n3=3, then one possible outcome is [(g1, g2), (g1,g4), (g2, g4)]

    returns
        3d nd.arrays of combines images.
    """
    random_idx = np.random.randint(0, len(imgs)-1, n1*n2).reshape(n2, n1)
    combined_imgs = []
    for idx in random_idx:
        img_list = np.array([imgs[i] for i in idx])
        combined_img = np.clip(img_list.sum(axis=0), 0, 1)
        combined_imgs.append(combined_img)
    return np.array(combined_imgs)


def filter_imgs(img):
    """
    filter objects that are too outlier.
    """
    too_big   = np.mean(img) > 0.85
    too_small = np.mean(img) < 0.1
    contains_full_row = np.any(np.min(img, axis=0))
    contains_full_col = np.any(np.min(img, axis=1))
    return not (too_big or too_small or contains_full_row or contains_full_col)