from data import generate_data
from utils import to_scipy_sdf

n_obj = 50
img_resolution = 20
save_name = "all"
save_name += str(img_resolution)
scipy = True
save_name += ""# "_test"
n_sample = 2000
obj_list = ("Circle", "nGon", "Rectangle", "Diamond", "CrossX")
# generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, n_sample=n_sample, save_name=save_name)

if scipy:
    to_scipy_sdf(save_name)


