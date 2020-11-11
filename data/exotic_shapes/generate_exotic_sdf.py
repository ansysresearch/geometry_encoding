import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.ndimage import distance_transform_edt
from data.utils import plot_sdf


def generate_exotic_sdf(size=100, plot=False):
    """
    run this file to generate sdf for exotic shapes, which are located
    in the png_images folder. There should not be any other file in that folder.

    :param size: sdf resolution. all images will be down-sampled to this resolution.
    :param plot: if True, img-sdf pairs are plotted.
    """
    import os
    folder_name = "data/exotic_shapes/png_images"
    print(os.getcwd())
    img_names = [join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
    sdf_vec = []
    for img_name in img_names:
        img = cv2.imread(img_name)

        # convert color images to gray
        if np.ndim(img) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert color images to black and white
        if len(np.unique(img)) > 2:
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        # invert and scale the image.
        img = (255 - img) / 255

        # make the image square.
        w, h = img.shape[0], img.shape[1]
        if w > h:
            l = (w - h) // 2
            r = w - h - l
            img = np.concatenate([np.zeros((w, l)), img, np.zeros((w, r))], axis=1)
        elif w < h:
            u = (h - w) // 2
            d = h - w - u
            img = np.concatenate([np.zeros((u, h)), img, np.zeros((d, h))], axis=0)

        # down sample the image
        img = cv2.resize(img, (size, size))
        img[img < 0.5] = 0
        img[img >= 0.5] = 1
        assert np.all(np.unique(img) == [0., 1.])

        # compute sdf
        sdf = -distance_transform_edt(img) + distance_transform_edt(1 - img)
        sdf /= (size // 2)
        sdf_vec.append(sdf)

        # plot
        if plot:
            plot_sdf(img, sdf)

    sdf_vec = np.array(sdf_vec)
    np.save("data/exotic_shapes/exotic_shapes" + str(size) + ".npy", sdf_vec)


if __name__ == "__main__":
    generate_exotic_sdf(size=128, plot=True)
