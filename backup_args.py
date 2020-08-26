# # training arg parser
# default_name = time.ctime().replace(" ", "_").replace(":", "")
# training_parser = argparse.ArgumentParser()
# training_parser.add_argument("-nw", "--network-id", type=int, help="network id",       default=1)
# training_parser.add_argument("-s",  "--save-name",  type=str, help="save name",        default=default_name)
# training_parser.add_argument("-d",  "--dataset-id", type=str, help="dataset id",       default="1")
# training_parser.add_argument("-ne", "--num-epochs", type=int, help="number of epochs", default=100)
# training_parser.add_argument("-se", "--save-every", type=int, help="save every",       default=10)
# training_parser.add_argument("-b",  "--batch-size", type=int, help="batch size",       default=50)
# training_parser.add_argument("-vf", "--val-frac",   type=float, help="validation fraction", default=0.1)
# training_parser.add_argument("-lr", "--learning-rate", type=float, help="learning rate", default=0.01)
# training_args = training_parser.parse_args()
#
# ## parsing input parameters
# network_id     = training_args.network_id
# save_name      = training_args.save_name
# dataset_id     = training_args.dataset_id
# num_epochs     = training_args.num_epochs
# save_every     = training_args.save_every
# batch_size     = training_args.batch_size
# lr             = training_args.learning_rate
# val_frac       = training_args.val_frac


# prediction_parser = argparse.ArgumentParser()
# prediction_parser.add_argument("-n",     "--network-id",   type=int, help="network id")
# prediction_parser.add_argument("-nfile", "--network-file", type=str, help="save name")
# prediction_parser.add_argument("-d",     "--data-name",    type=str, help="data name")
# prediction_parser.add_argument("-p",     "--plot-arg",     type=int,
#                                help="0 for no plot, 1 for predict and plot, 2 for plot only (read from file)",
#                                default=0)
#
# prediction_args = prediction_parser.parse_args()
# network_id = prediction_args.network_id
# network_file = prediction_args.network_file
# data_name = prediction_args.data_name
# plot_arg = prediction_args.plot_arg


# obj_list_default = ("Circle", "Rectangle", "Diamond", "CrossX", "nGon")
# datagen_parser = argparse.ArgumentParser()
# datagen_parser.add_argument("-obj", "--obj-list",  type=str,  help="objects in the data", default=obj_list_default)
# datagen_parser.add_argument("-num", "--n-obj",     type=int,  help="number of data",      default=500)
# datagen_parser.add_argument("-sav", "--save-name", type=str,  help="save name",           default="all")
# datagen_parser.add_argument("-res", "--img-res",   type=int,  help="image resolution",    default=100)
# datagen_args = datagen_parser.parse_args()
# n_obj = datagen_args.n_obj
# img_resolution = datagen_args.img_res
# save_name = datagen_args.save_name
# obj_list = datagen_args.obj_list
# #save_name += "-test"
# n_sample = 2000
# obj_list = ("Circle", "nGon", "Rectangle", "Diamond", "CrossX")
