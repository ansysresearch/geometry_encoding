import numpy as np
from network import get_network

from params import *
from utils import read_data

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


device = 'cpu'
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
net.eval()

train_ds, _ = read_data(dataset_id, val_frac=0)
train_data = compute_prediction(train_ds)
train_prediction_file_name = prediction_save_dir + "train_predictions_" + save_name + ".npy"
np.save(train_prediction_file_name, train_data)

test_ds, _ = read_data(dataset_id, val_frac=0, end_suffix="_test")
test_data = compute_prediction(test_ds)
test_prediction_file_name = prediction_save_dir + "test_predictions_" + save_name + ".npy"
np.save(test_prediction_file_name, test_data)
