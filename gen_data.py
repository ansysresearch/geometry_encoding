from data import generate_data

n_obj = 300
img_resolution = 50
save_name = "circle"
save_name += str(img_resolution)
#save_name += "-test"
n_sample = 1000
obj_list = ("Circle",)# "nGon", "Rectangle", "Diamond", "CrossX")

generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, n_sample=n_sample, save_name=save_name)

