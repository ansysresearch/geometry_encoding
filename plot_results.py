from params import *
from utils import plot_prediction_results, plot_interpolation_results

dtype      = DATA_TYPE
network_id = NETWORK_ID
dataset_id = DATASET_ID
save_name  = NETWORK_SAVE_NAME
n_prediction     = COMPUTE_PREDICTIONS_NUM
network_save_dir = NETWORK_SAVE_DIRECTORY
prediction_save_dir = PREDICTION_SAVE_DIRECTORY

train_prediction_file_name = prediction_save_dir + "train_predictions_" + save_name + ".npy"
plot_prediction_results(train_prediction_file_name)

test_prediction_file_name = prediction_save_dir + "test_predictions_" + save_name + ".npy"
plot_prediction_results(test_prediction_file_name)

test_prediction_exotic_shapes_file_name = prediction_save_dir + "test_predictions_exotic_shapes_" + save_name + ".npy"
plot_prediction_results(test_prediction_exotic_shapes_file_name)

test_interpolation_file_name = prediction_save_dir + "test_interpolate_data" + save_name + ".npy"
plot_interpolation_results(test_interpolation_file_name)