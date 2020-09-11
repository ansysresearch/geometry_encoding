import pickle
import numpy as np
from network import get_network

from params import *
from utils import read_data, compute_perimeter_img

dtype      = DATA_TYPE
network_id = NETWORK_ID
dataset_id = DATASET_ID
save_name  = NETWORK_SAVE_NAME
n_prediction     = COMPUTE_PREDICTIONS_NUM
network_save_dir = NETWORK_SAVE_DIRECTORY
prediction_save_dir = PREDICTION_SAVE_DIRECTORY


def compute_prediction(ds):
    saved_list = []
    for idx in np.random.permutation(len(ds))[:n_prediction]:
        img, sdf = ds[idx]
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=dtype)
        with torch.no_grad():
            sdf_pred = net(img)
        sdf = sdf.squeeze()
        sdf_pred = sdf_pred.squeeze()
        saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))
    return np.array(saved_list)


def compute_accuracy_metrics(test_preds):
    sdf_error_means = np.mean(np.diff(test_preds, axis=1), axis=(2, 3))
    sdf_error_stds = np.std(np.diff(test_preds, axis=1), axis=(2, 3))
    imgs = (test_preds < 0).astype(int)
    area = np.mean(imgs, axis=(2, 3))
    area_error = np.diff(area, axis=1) / area[:, 0]
    perimeter_error = np.zeros(imgs.shape[0],)
    for i in range(imgs.shape[0]):
        true_perimeter = compute_perimeter_img(imgs[i, 0, :, :])
        pred_perimeter = compute_perimeter_img(imgs[i, 1, :, :])
        perimeter_error[i] = (true_perimeter - pred_perimeter) / true_perimeter

    all_results = {"sdf_error_means": sdf_error_means,
                   "sdf_error_stds": sdf_error_stds,
                   "area_error": area_error,
                   "perimeter_error": perimeter_error}
    return all_results


# importing network
device = 'cpu'
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
net.eval()

# evaluating on train dataset
train_ds, _ = read_data(dataset_id, val_frac=0)
train_data = compute_prediction(train_ds)
train_prediction_file_name = prediction_save_dir + "train_predictions_" + save_name + ".npy"
np.save(train_prediction_file_name, train_data)

# evaluating on test dataset
test_ds, _ = read_data(dataset_id, val_frac=0, end_suffix="_test")
test_data = compute_prediction(test_ds)
test_prediction_file_name = prediction_save_dir + "test_predictions_" + save_name + ".npy"
np.save(test_prediction_file_name, test_data)

# train_results_dict = compute_accuracy_metrics(train_data)
# train_error_dict_file_name = prediction_save_dir + "train_error_dict_" + save_name + ".pickle"
# with open(train_error_dict_file_name, 'wb') as f:
#     pickle.dump(train_results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# test_results_dict = compute_accuracy_metrics(test_data)
# test_error_dict_file_name = prediction_save_dir + "test_error_dict_" + save_name + ".pickle"
# with open(test_error_dict_file_name, 'wb') as f:
#     pickle.dump(test_results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
