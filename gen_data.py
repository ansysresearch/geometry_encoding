from data import generate_data_on_grid
import argparse


n_obj    = 500
augment  = False
save_name = "rect-circ-50"
img_resolution = 50
obj_list = ["Circle", "Rectangle"]

generate_data_on_grid(obj_list=obj_list, n_obj=n_obj, augment=augment, save_name=save_name, img_resolution=img_resolution)