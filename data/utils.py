import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def compute_numerical_sdf(img):
    """
    compute SDF of a given img using two distance transformation
    see https://stackoverflow.com/questions/44770396/how-does-the-scipy-distance-transform-edt-function-work
    Args
        img (np.ndarray) 2d black and white object:
    """
    s = img.shape[0]
    dist1, indices1 = distance_transform_edt(img, return_indices=True)
    dist2, indices2 = distance_transform_edt(1-img, return_indices=True)
    sdf_numerical = dist2 - dist1
    sdf_numerical /= (s // 2)
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
