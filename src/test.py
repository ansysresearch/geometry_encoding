import os
import torch
import numpy as np
from src import read_data, get_save_name, get_dtype, get_network, viz


def compute_prediction(args, ds):
    dtype = get_dtype(args)
    network_id = args.net_id
    save_name = get_save_name(args)
    n_prediction = args.n_pred
    checkpoint_dir = args.ckpt_dir
    network_save_dir = os.path.join(checkpoint_dir, 'networks')

    # importing network
    device = 'cpu'
    net = get_network(network_id=network_id).to(device=device, dtype=dtype)
    net.load_state_dict(torch.load(os.path.join(network_save_dir, save_name + ".pth"), map_location=device))
    net.eval()

    saved_list = []
    for idx in np.random.permutation(len(ds))[:n_prediction]:
        xb, yb = ds[idx]
        xb = xb.unsqueeze(0).to(device=device, dtype=dtype)
        yb = yb.to(device=device, dtype=dtype)
        with torch.no_grad():
            yb_pred = net(xb)

        yb, yb_pred = yb.squeeze(), yb_pred.squeeze()
        saved_list.append((yb.numpy().copy(), yb_pred.numpy().copy()))
    return np.array(saved_list)


def test(args):
    save_name = get_save_name(args)
    prediction_save_dir = os.path.join(args.ckpt_dir, 'predictions')

    # evaluating on train dataset
    train_ds, _ = read_data(args)
    train_data = compute_prediction(args, train_ds)
    train_prediction_file_name = os.path.join(prediction_save_dir, "train_predictions_" + save_name + ".npy")
    np.save(train_prediction_file_name, train_data)

    # evaluating on test dataset
    test_ds, _ = read_data(args, end_suffix="_test")
    test_data = compute_prediction(args, test_ds)
    test_prediction_file_name = os.path.join(prediction_save_dir, "test_predictions_" + save_name + ".npy")
    np.save(test_prediction_file_name, test_data)

    # evaluating on exotic dataset
    exotic_ds, _ = read_data(args, end_suffix="_exotic")
    exotic_data = compute_prediction(args, exotic_ds)
    exotic_prediction_file_name = os.path.join(prediction_save_dir, "exotic_predictions_" + save_name + ".npy")
    np.save(exotic_prediction_file_name, exotic_data)

    viz(args)
