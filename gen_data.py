from data import generate_data
import argparse

obj_list_default = ("Circle", "Rectangle", "Diamond", "Cross", "nGon")
datagen_parser = argparse.ArgumentParser()
datagen_parser.add_argument("-obj", "--obj-list",  type=str,  help="objects in the data", default=obj_list_default)
datagen_parser.add_argument("-n",   "--n-obj",     type=int,  help="number of data",      default=500)
datagen_parser.add_argument("-a",   "--augment",   type=bool, help="augment data",        default=1)
datagen_parser.add_argument("-s",   "--save-name", type=str,  help="save name",           default="train")
datagen_parser.add_argument("-r",   "--img-res",   type=int,  help="image resolution",    default=100)
datagen_args = datagen_parser.parse_args()

obj_list = datagen_args.obj_list.split(" ")
n_obj    = datagen_args.n_obj
augment  = datagen_args.augment
save_name = datagen_args.save_name
img_resolution = datagen_args.img_res

generate_data(obj_list=obj_list, n_obj=n_obj, augment=augment, save_name=save_name, img_resolution=img_resolution)