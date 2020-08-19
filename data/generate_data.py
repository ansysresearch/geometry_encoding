import tqdm
from data.geoms import *


def generate_geoms(obj_list, xmax=0.8, ymax=0.8):
    obj_id = np.random.choice(obj_list)
    #print(obj_id)
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
    elif obj_id == "Cross":
        w = max(0.2, np.random.random() * xmax)
        r = max(0.2, np.random.random() * xmax*0.3)
        geom = CrossX([w, r])
    else:
        raise("object %s is not yet implemented"%obj_id)
    return geom


def augment_geom(geom, mode="all"):
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


def combine_sdf(sdfs_one_object, n_combine, mode="union"):
    sdf_random_list = np.random.choice(sdfs_one_object, n_combine)
    if mode == "union":
        return np.min(sdf_random_list, axis=0)
    else:
        raise(NotImplementedError("only mode union is implemented."))


def generate_data(img_resolution=200, n_obj=500, augment=True, save_name=None, plot=False,
                  obj_list=("Circle", "Rectangle", "Diamond", "Cross", "nGon")):
    x, y = np.meshgrid(np.linspace(-1, 1, img_resolution), np.linspace(-1, 1, img_resolution))
    print("generating objects")
    geoms = []
    for _ in tqdm.tqdm(range(n_obj)):
        geoms.append(generate_geoms(obj_list))

    print("augmenting with rotation, translation and scaling")
    geoms_aug = []
    for _ in range(3):
        geoms_aug += [augment_geom(augment_geom(g, mode="translate")) for g in geoms]

    sdf = [g.eval_sdf(x, y) for g in geoms_aug]
    if augment:
        n_two_obj = n_obj * 2
        n_three_obj = n_obj * 3
        print("augment with merging")
        sdf += [combine_sdf(sdf, 2) for _ in range(n_two_obj)] + [combine_sdf(sdf, 3) for _ in range(n_three_obj)]

    sdf = np.array(sdf)

    if save_name:
        print("save data")
        np.save("datasets/X_" + save_name +".npy", sdf < 0)
        np.save("datasets/Y_" + save_name +".npy", sdf)

    if plot:
        for idx in np.random.randint(0, sdf.shape[0], 10):
            plot_sdf(sdf[idx, :, :] < 0, sdf[idx, :, :])