import numpy as np
from network import get_network
from params import *

dtype      = DATA_TYPE
network_id = NETWORK_ID
dataset_id = "data/exotic_shapes/exotic_shapes" + str(IMAGE_RESOLUTION) + ".npy"
save_name  = NETWORK_SAVE_NAME
n_prediction   = COMPUTE_PREDICTIONS_NUM
network_save_dir = NETWORK_SAVE_DIRECTORY
prediction_save_dir = PREDICTION_SAVE_DIRECTORY

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

    img = img.squeeze()
    sdf = sdf.squeeze()
    sdf_pred = sdf_pred.squeeze()

    saved_list.append([sdf.numpy().copy(), sdf_pred.numpy().copy()])

saved_list = np.array(saved_list)
test_prediction_exotic_shapes_file_name = prediction_save_dir + "test_predictions_exotic_shapes_" + save_name + ".npy"
np.save(test_prediction_exotic_shapes_file_name, saved_list)
