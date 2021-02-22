import os
import torch
import numpy as np
from src.network import get_network
from src import read_data, get_save_name, get_dtype, get_device


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

    # train_results_dict = compute_accuracy_metrics(train_data)
    # train_error_dict_file_name = prediction_save_dir + "train_error_dict_" + save_name + ".pickle"
    # with open(train_error_dict_file_name, 'wb') as f:
    #     pickle.dump(train_results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # test_results_dict = compute_accuracy_metrics(test_data)
    # test_error_dict_file_name = prediction_save_dir + "test_error_dict_" + save_name + ".pickle"
    # with open(test_error_dict_file_name, 'wb') as f:
    #     pickle.dump(test_results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


# def test_exomodel_flag



# def compute_accuracy_metrics(test_preds):
#     sdf_error_means = np.mean(np.diff(test_preds, axis=1), axis=(2, 3))
#     sdf_error_stds = np.std(np.diff(test_preds, axis=1), axis=(2, 3))
#     imgs = (test_preds < 0).astype(int)
#     area = np.mean(imgs, axis=(2, 3))
#     area_error = np.diff(area, axis=1) / area[:, 0]
#     perimeter_error = np.zeros(imgs.shape[0],)
#     for i in range(imgs.shape[0]):
#         true_perimeter = compute_perimeter_img(imgs[i, 0, :, :])
#         pred_perimeter = compute_perimeter_img(imgs[i, 1, :, :])
#         perimeter_error[i] = (true_perimeter - pred_perimeter) / true_perimeter
#
#     all_results = {"sdf_error_means": sdf_error_means,
#                    "sdf_error_stds": sdf_error_stds,
#                    "area_error": area_error,
#                    "perimeter_error": perimeter_error}
#     return all_results