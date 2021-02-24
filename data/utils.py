import numpy as np
# from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance_matrix
from scipy.interpolate import interp2d


def compute_numerical_sdf(img, return_random_points=False):
    s = img.shape[0]
    dist1, indices1 = distance_transform_edt(img, return_indices=True)
    dist2, indices2 = distance_transform_edt(1-img, return_indices=True)
    sdf_numerical = dist2 - dist1
    sdf_numerical /= (s // 2)
    if return_random_points:
        n_random_points = 2000
        x = np.linspace(-1, 1, s)
        random_points = np.random.random(n_random_points) * 2 - 1
        random_points = np.sort(random_points)
        f_interp = interp2d(x, x, sdf_numerical, kind='cubic')
        random_points_sdf = f_interp(random_points, random_points)
        xidx = np.random.randint(0, n_random_points, n_random_points)
        yidx = np.random.randint(0, n_random_points, n_random_points)
        random_points = np.array([random_points[xidx], random_points[yidx], random_points_sdf[yidx, xidx]]).T
        # indices_out = np.unique(indices1[:, img > 0], axis=1)
        # boundary_out = np.array([x[indices_out[1, :]], x[indices_out[0, :]]])
        # indices_in = np.unique(indices2[:, img < 1], axis=1)
        # boundary_in = np.array([x[indices_in[1, :]], x[indices_in[0, :]]])
        # dist_in = distance_matrix(random_points, boundary_in.T).min(axis=1)
        # dist_out = distance_matrix(random_points, boundary_out.T).min(axis=1)
        # random_points_sdf = - dist_out
        # random_points_sdf[dist_in > dist_out] = dist_in[dist_in > dist_out]
        # random_points_sdf = []
        # for point in random_points:
        #     dist_in  = np.linalg.norm(boundary_in - point.T, axis=0).min()
        #     dist_out = np.linalg.norm(boundary_out - point.T, axis=0).min()
        #     if dist_in < dist_out:
        #         random_points_sdf.append(-dist_out)
        #     else:
        #         random_points_sdf.append(dist_in)
        # random_points_sdf = random_points_sdf.reshape(-1, 1)
        # random_points = np.concatenate([random_points, random_points, random_points_sdf], axis=1)
        return sdf_numerical, random_points
    else:
        return sdf_numerical


def plot_sdf(img, sdf, xticks=(-1, 1), yticks=(-1, 1), plot_eikonal=False, show=True, colorbar=False):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='binary')
    plt.gca().invert_yaxis()
    plt.xticks(np.linspace(0, img.shape[0], 5), np.linspace(xticks[0], xticks[1], 5))
    plt.yticks(np.linspace(0, img.shape[1], 5), np.linspace(yticks[0], yticks[1], 5))

    plt.subplot(2, 2, 2)
    plt.imshow(sdf, cmap='hot')
    if colorbar: plt.colorbar()
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