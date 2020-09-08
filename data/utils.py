import numpy as np
#from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def plot_sdf(img, sdf, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False, show=True):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='binary')
    plt.gca().invert_yaxis()
    plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
    plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))

    plt.subplot(2, 2, 2)
    plt.imshow(sdf, cmap='hot')
    #plt.colorbar()
    plt.contour(sdf, 30, colors='k')
    plt.gca().invert_yaxis()
    plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
    plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))

    if plot_eikonal:
        dx, dy = (xticks[1] - xticks[0]) / img.shape[0],  (yticks[1] - yticks[0]) / img.shape[1]
        sdf_gradient = np.gradient(sdf)
        sdf_gradient_values = np.sqrt(sdf_gradient[0] ** 2 / dx ** 2 + sdf_gradient[1] ** 2 / dy ** 2)

        plt.subplot(2, 2, 3)
        plt.imshow(sdf_gradient_values)
        plt.gca().invert_yaxis()
        plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
        plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))
        plt.colorbar()

    if show:
        plt.show()


def plot_ply(ply):
    """
    Plot vertices and triangles from a PlyData instance.
    Assumptions:
        `ply' has a 'vertex' element with 'x', 'y', and 'z'
            properties;
        `ply' has a 'face' element with an integral list property
            'vertex_indices', all of whose elements have length 3.
    """
    vertex = ply['vertex']

    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))

    mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

    if 'face' in ply:
        tri_idx = ply['face']['vertex_indices']
        triangles = np.vstack(tri_idx)
        mlab.triangular_mesh(x, y, z, triangles,
                             color=(1, 0, 0.4), opacity=0.5)
    mlab.show()


def to_scipy_sdf(file_name):
    imgs = np.load("datasets/img_" + file_name + ".npy")
    scipy_sdfs = []
    for img in imgs:
        s1, s2 = img.shape
        assert s1 == s2
        assert np.all(np.unique(img) == [0., 1.])
        scipy_sdf = -distance_transform_edt(img) + distance_transform_edt(1-img)
        scipy_sdf /= (s1 // 2)
        scipy_sdfs.append(scipy_sdf)
    scipy_sdfs = np.array(scipy_sdfs)
    np.save("datasets/sdf_scipy_" + file_name + ".npy", scipy_sdfs)


if __name__ == "__main__":
    to_scipy_sdf("all128")
