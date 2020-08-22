import tqdm
import itertools
from scipy.optimize import fsolve
from scipy.ndimage import center_of_mass
from data.geoms import *


def generate_one_geometry(obj_list, xmax=0.8, ymax=0.8):
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


def augment_one_geometry(geom, mode="all"):
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


def generate_geometries(n_obj=500, n_aug=3, obj_list=("Circle", "Rectangle", "Diamond", "Cross", "nGon")):

    print("generating objects")
    geoms = [generate_one_geometry(obj_list) for _ in range(n_obj)]

    # geoms is centered at origin. we randomly translate all geoms
    geoms = [augment_one_geometry(g, mode="translate") for g in geoms]

    # now produce replicates of geoms with random rotation, scaling or translation
    geoms_aug = []
    print("augmenting with rotation, translation and scaling")
    for _ in range(n_aug):
        geoms_aug += [augment_one_geometry(g) for g in geoms]
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


def sample_near_geometry(geoms, n_sample, lb=-0.1, ub=0.1):
    sample_pnts = []
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


def generate_data(obj_list, img_resolution=100, n_obj=500, save_name="data", plot=False, n_sample=500):
    x, y = np.meshgrid(np.linspace(-1, 1, img_resolution), np.linspace(-1, 1, img_resolution))
    geoms1 = generate_geometries(n_obj=n_obj, obj_list=obj_list)
    sdf1 = [g.eval_sdf(x, y) for g in geoms1]
    geoms2, sdf2 = combine_geometries(geoms1, 2, 2*n_obj, x, y)
    geoms3, sdf3 = combine_geometries(geoms1, 3, 3*n_obj, x, y)
    sdf = np.array(sdf1 + sdf2 + sdf3)

    print("save sdf on grids")
    np.save("data/datasets/img_" + save_name + ".npy", sdf < 0)
    np.save("data/datasets/sdf_" + save_name + ".npy", sdf)

    print("sampling points on the boundary")
    sample1 = [sample_near_geometry(geom, int(n_sample*0.2), lb=-0.01, ub=0.01) for geom in geoms1 + geoms2 + geoms3]

    print("sampling points near the boundary")
    sample2 = [sample_near_geometry(geom, int(n_sample*0.2), lb=-0.1, ub=0.1) for geom in geoms1 + geoms2 + geoms3]

    print("sampling points anywhere else")
    sample3 = [sample_near_geometry(geom, int(n_sample*0.6), lb=-2, ub=5) for geom in geoms1 + geoms2 + geoms3]
    sample = np.concatenate([sample1, sample2, sample3], axis=2)

    print("save sampling points")
    np.save("data/datasets/sample_" + save_name + ".npy", sdf)

    if plot:
        for idx in np.random.randint(0, sdf.shape[0], 50):
            sd, sp = sdf[idx, :, :], sample[idx, :, :]
            plot_sdf(sd, sd < 0, show=False)
            plt.plot((sp[0, :] + 1) * sd.shape[1]/2, (sp[1, :] + 1) * sd.shape[0]/2, 'g.')
            # plt.plot((sp[0, 400:] + 1) * sd.shape[1]/2, (sp[1, 400:] + 1) * sd.shape[0]/2, 'g.')
            # plt.plot((sp[0, 200:400] + 1) * sd.shape[1]/2, (sp[1, 200:400] + 1) * sd.shape[0]/2, 'b.')
            # plt.plot((sp[0, :200] + 1) * sd.shape[1]/2, (sp[1, :200] + 1) * sd.shape[0]/2, 'r.')
            plt.show()


if __name__ == "__main__":
    generate_data(["Circle", "nGon", "Rectangle"], img_resolution=100, n_obj=5, n_sample=1000, save_name="data", plot=True)



