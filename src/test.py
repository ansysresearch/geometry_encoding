import torch
import numpy as np
from src.network import get_network
from src import read_data, compute_perimeter_img


def compute_prediction(args, ds):
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    network_id = args.net_id
    save_name = args.net_id + '_' + args.dataset_id + str(args.img_res)
    n_prediction = args.n_pred
    checkpoint_dir = args.ckpt_dir
    network_save_dir = checkpoint_dir + 'networks/'

    # importing network
    device = 'cpu'
    net = get_network(network_id=network_id).to(device=device, dtype=dtype)
    net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
    net.eval()

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


def test(args):
    save_name = args.net_id + '_' + args.dataset_id + str(args.img_res)
    prediction_save_dir = args.ckpt_dir + 'predictions/'

    # evaluating on train dataset
    train_ds, _ = read_data(args)
    train_data = compute_prediction(args, train_ds)
    train_prediction_file_name = prediction_save_dir + "train_predictions_" + save_name + ".npy"
    np.save(train_prediction_file_name, train_data)

    # evaluating on test dataset
    test_ds, _ = read_data(args, end_suffix="_test")
    test_data = compute_prediction(args, test_ds)
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


def test_exotic_shape(args):
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    network_id = args.net_id
    dataset_id = "data/exotic_shapes/exotic_shapes" + str(args.img_res) + ".npy"
    save_name = args.net_id + '_' + args.dataset_id + str(args.img_res)
    checkpoint_dir = args.ckpt_dir
    network_save_dir = checkpoint_dir + 'networks/'
    prediction_save_dir = checkpoint_dir + 'predictions/'

    device = 'cpu'
    net = get_network(network_id=network_id).to(device=device, dtype=dtype)
    net.load_state_dict(torch.load(network_save_dir + save_name + ".pth", map_location=device))
    net.eval()

    X = np.load(dataset_id).astype(float)
    img_resolution = X.shape[-1]
    X = X.reshape((-1, 1, img_resolution, img_resolution))
    Y = X.copy()
    X = (X < 0).astype(float)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    saved_list = []
    for img, sdf in zip(X, Y):
        img = img.unsqueeze(0)
        sdf = sdf.unsqueeze(0)
        img = img.to(device=device, dtype=dtype)

        with torch.no_grad():
            sdf_pred = net(img)

        sdf = sdf.squeeze()
        sdf_pred = sdf_pred.squeeze()

        saved_list.append([sdf.numpy().copy(), sdf_pred.numpy().copy()])

    saved_list = np.array(saved_list)
    test_prediction_exotic_shapes_file_name = prediction_save_dir + "test_predictions_exotic_shapes_" + save_name + ".npy"
    np.save(test_prediction_exotic_shapes_file_name, saved_list)




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