import torch
import numpy as np
from network import get_network
from utils import plot_data, read_data

plot_arg = 1

if plot_arg == 2:
    data_file = "checkpoints/train_predictions_UNet_all256.npy"
    plot_data(data_file)
else:
    network_id = "UNet"
    dataset_id = "all256"
    save_name  = network_id + "_" + dataset_id #+ "_g20"
    def compute_prediction(ds):
        saved_list = []
        for idx in np.random.randint(0, len(ds), 10):
            img, sdf = ds[idx]
            img = img.unsqueeze(0)
            img = img.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                sdf_pred = net(img)
            sdf = sdf.squeeze()
            sdf_pred = sdf_pred.squeeze()
            saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))
        return np.array(saved_list)


    device = 'cpu'
    net = get_network(network_id=network_id).to(device=device)
    #net.load_state_dict(torch.load("checkpoints/" + save_name + ".pth"))
    net.load_state_dict(torch.load("checkpoints/" + save_name + ".pth", map_location=device))
    net.eval()

    train_ds, _ = read_data(dataset_id, val_frac=0)
    train_data = compute_prediction(train_ds)
    train_prediction_file_name = "checkpoints/train_predictions_" + save_name + ".npy"
    np.save(train_prediction_file_name, train_data)

    test_ds, _ = read_data(dataset_id, val_frac=0, end_suffix="_test")
    test_data = compute_prediction(test_ds)
    test_prediction_file_name = "checkpoints/test_predictions_" + save_name + ".npy"
    np.save(test_prediction_file_name, test_data)

    if plot_arg == 1:
        plot_data(train_prediction_file_name)
        plot_data(test_prediction_file_name)
