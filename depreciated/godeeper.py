import torch
import numpy as np
from network import get_network
from utils import plot_data, read_data
import matplotlib.pyplot as plt

network_id = "UNet"
dataset_id = "all128"
save_name  = "UNet_all128"
dtype = torch.float32

device = 'cpu'
net = get_network(network_id=network_id).to(device=device, dtype=dtype)
net.load_state_dict(torch.load("checkpoints/" + save_name + ".pth", map_location=device))
net.eval()

train_ds, _ = read_data(dataset_id, val_frac=0)
idx = 0
img, sdf = train_ds[idx]
img = img.unsqueeze(0)
img = img.to(device=device, dtype=dtype)
with torch.no_grad():
    inc_val = net.inc(img)
    down1_val = net.down1(inc_val)
    down2_val = net.down2(down1_val)
    down3_val = net.down3(down2_val)
    down4_val = net.down4(down3_val)
    up1 = net.up1(down4_val, down3_val)
    up2 = net.up2(up1, down2_val)
    up3 = net.up3(up2, down1_val)
    up4 = net.up4(up3, inc_val)
    sdf_pred = net.outc(up4)


def grad_mag(sdf):
    dx, dy = 2 / sdf.shape[0],  2 / sdf.shape[1]
    sdf_gradient = np.gradient(sdf)
    sdf_gradient_magnitude = np.sqrt(sdf_gradient[0] ** 2 / dx ** 2 + sdf_gradient[1] ** 2 / dy ** 2)
    return sdf_gradient_magnitude


plt.figure(figsize=(12, 10))
plt.subplot(2, 3, 1)
plt.imshow(img.squeeze(), cmap='binary')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(sdf.squeeze(), cmap='hot')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(sdf_pred.squeeze(), cmap='hot')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 3, 4)
plt.imshow(sdf.squeeze() - sdf_pred.squeeze(), cmap='hot')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(grad_mag(sdf.squeeze()) - 1, cmap='hot')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 3, 6)
plt.imshow(grad_mag(sdf_pred.squeeze()) - 1, cmap='hot')
plt.gcf().suptitle("grad magnitude")
plt.colorbar()
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.show()

plt.show()

plt.figure(figsize=(12, 12))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(inc_val.squeeze()[i, :, :], cmap='hot')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
plt.gcf().suptitle("first 64 conv channels")
plt.show()


plt.figure(figsize=(12, 12))
for i in range(256):
    plt.subplot(16, 16, i+1)
    plt.imshow(up1.squeeze()[i, :, :], cmap='hot')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
plt.show()

plt.figure(figsize=(12, 12))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(up3.squeeze()[i, :, :], cmap='hot')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
plt.gcf().suptitle("up 3")
plt.show()


plt.figure(figsize=(12, 12))
for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(up4.squeeze()[i, :, :], cmap='hot')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
plt.gcf().suptitle("last layer before final conv")
plt.show()




plt.subplot(2, 1, 1)
plt.imshow(grad_mag(sdf.squeeze()), cmap='hot')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(grad_mag(sdf_pred.squeeze()), cmap='hot')
plt.gcf().suptitle("grad magnitude")
plt.colorbar()
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.show()



