import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def get_edges(img, edge_lower_tresh=100, edge_upper_tresh=200):
    return cv2.Canny(img, edge_lower_tresh, edge_upper_tresh)


def get_area(img_bw, scale=1.):
    area = np.sum(img_bw) / img_bw.max()
    #area /= np.prod(img_bw.shape)
    area *= (scale ** 2)
    return area


def get_perimeter(img_bw, scale=1.):
    edges = get_edges(img_bw)
    perimeter = np.sum(edges) / img_bw.max()
    #perimeter /= img_bw.shape[0]
    perimeter *= scale
    return perimeter


def to_bw_img(img_gray, bw_lower_tresh=200, bw_upper_tresh=255):
    img_bw = cv2.threshold(img_gray, bw_lower_tresh, bw_upper_tresh, cv2.THRESH_BINARY)[1]
    return img_bw


def plot_img(img, img_boxes=None, figsize=(10,10), axis=False, title=None):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if np.ndim(img) == 2: plt.set_cmap("gray")
    if not axis: plt.axis("off")
    if title: plt.title(title)
    if img_boxes is not None:
        rects = [Rectangle((img_box.left, img_box.top), img_box.width, img_box.height) for img_box in img_boxes]
        pc = PatchCollection(rects, edgecolor='r', facecolors='none', linewidths=2)
        plt.gca().add_collection(pc)
    plt.show()


from data.geoms import *
geom = Circle(0.3)
x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
sdf = geom.eval_sdf(x, y)
img = sdf < 0
# img = np.zeros((128, 128))
# img[30:50, 40:80] = 255
img = img.astype(np.uint8) * 255
img_bw = to_bw_img(img)
plot_img(img_bw)
edges = get_edges(img_bw)
plot_img(edges)
sobelx = cv2.Sobel(img_bw, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img_bw, cv2.CV_64F, 0, 1, ksize=5)
plot_img(sobelx)
p = get_perimeter(img_bw)#, scale=2/128)
a = get_area(img_bw)#, scale=2/128)
print(p, " ", a)
print("")
