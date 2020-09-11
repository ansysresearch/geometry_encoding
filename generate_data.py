from params import *
from data import generate_data

n_obj = NUMBER_OBJECTS
img_resolution = IMAGE_RESOLUTION
save_name = DATASET_ID
obj_list = ("Circle", "nGon", "Rectangle", "Diamond")

# train set
generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, save_name=save_name)

# test set
n_obj_test = n_obj // 10
save_name_test = save_name + "_test"
generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj_test, save_name=save_name_test)
