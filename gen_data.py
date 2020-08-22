from data import generate_data

n_obj    = 30
augment  = False
img_resolution = 50
save_name = "circ"
save_name += str(img_resolution)
save_name += "-test"
n_sample = 1000
obj_list = ["Circle"]

generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, n_sample=1000, save_name=save_name)
