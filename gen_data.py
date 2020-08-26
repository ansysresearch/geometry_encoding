from data import generate_data
import argparse

n_obj = 30
img_resolution = 50
save_name = "all"
save_name += str(img_resolution)
save_name += "_test"
n_sample = 2000
obj_list = ("Circle", "nGon", "Rectangle", "Diamond", "CrossX")
generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, n_sample=n_sample, save_name=save_name)

