import torch
import numpy as np
from network import get_network
from utils import plot_data

dtype = torch.float32
network_id = "UNet6"
network_file = "UNet6_all128"
data_name = "data/exotic_shapes/exotic_shapes128.npy"

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
try:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth"))
except RuntimeError:
    net.load_state_dict(torch.load("checkpoints/" + network_file + ".pth", map_location=device))
net.eval()

X = np.load(data_name).astype(float)
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

    saved_list.append((sdf.numpy().copy(), sdf_pred.numpy().copy()))

save_list = np.array(saved_list)
test_prediction_exotic_shapes_file_name = "checkpoints/test_predictions_exotic_shapes_" + network_file + ".npy"
np.save(test_prediction_exotic_shapes_file_name, save_list)
plot_data(test_prediction_exotic_shapes_file_name)