from data import generate_data, generate_offgrid_data

n_obj = 10
img_resolution = 128
save_name = "all"
save_name += str(img_resolution)
scipy = True
save_name += "_test"
n_sample = 1000
obj_list = ("Circle", "nGon", "Rectangle", "Diamond")#, "CrossX")
imgs, sdfs = generate_data(obj_list, img_resolution=img_resolution, n_obj=n_obj, save_name=save_name)

#pnts = generate_offgrid_data(obj_list, img_resolution=img_resolution, n_obj=n_obj//10, save_name=save_name)
# from data.utils import plot_sdf
# print(pnts[0, :, :5])
# plot_sdf(imgs[0, :, :], sdfs[0, :, :], colorbar=True)
