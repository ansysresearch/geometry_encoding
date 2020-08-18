import tqdm
from data.geoms import *


def generate_geoms(obj_list=("Circle", "Rectangle", "Diamond", "Cross", "nGon"), xmax=0.8, ymax=0.8):
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
    idx = np.random.randint(0, sdfs_one_object.shape[0], n_combine)
    if mode == "union":
        return np.min(sdfs_one_object[idx], axis=0)
    else:
        raise(NotImplementedError("only mode union is implemented."))


n_one_obj, n_two_obj, n_three_obj, n_aug = 50, 1000, 2000, 3
grid_N = 200
x, y = np.meshgrid(np.linspace(-1, 1, grid_N), np.linspace(-1, 1, grid_N))

print("generating objects")
geoms = []
for _ in tqdm.tqdm(range(n_one_obj)):
    geoms.append(generate_geoms())

print("augmenting with rotation, translation and scaling")
geoms_aug = []
for _ in range(n_aug):
    geoms_aug += [augment_geom(augment_geom(g, mode="translate")) for g in geoms]

print("augment with merging")
sfds_one_object = np.array([g.eval_sdf(x, y) for g in geoms_aug])
sdfs_two_object = np.array([combine_sdf(sfds_one_object, 2) for _ in range(n_two_obj)])
sdfs_three_object = np.array([combine_sdf(sfds_one_object, 3) for _ in range(n_three_obj)])

print("save data")
np.save("X_1obj.npy", sfds_one_object < 0)
np.save("Y_1obj.npy", sfds_one_object)
np.save("X_2obj.npy", sdfs_two_object < 0)
np.save("Y_2ojb.npy", sdfs_two_object)
np.save("X_3obj.npy", sdfs_three_object < 0)
np.save("Y_3ojb.npy", sdfs_three_object)

plot_data = True # turn true for plotting
if plot_data:
    for i in np.random.randint(0, n_one_obj, 10):
        sdf = sfds_one_object[i, :, :]
        img = sdf < 0
        plot_sdf(img, sdf, plot_eikonal=True)

    for i in np.random.randint(0, n_two_obj, 10):
        sdf = sdfs_two_object[i, :, :]
        img = sdf < 0
        plot_sdf(img, sdf, plot_eikonal=True)

    for i in np.random.randint(0, n_three_obj, 10):
        sdf = sdfs_three_object[i, :, :]
        img = sdf < 0
        plot_sdf(img, sdf, plot_eikonal=True)
