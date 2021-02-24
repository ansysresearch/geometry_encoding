import torch
import numpy as np
from torch.autograd import Function
from scipy.interpolate import Rbf, RectBivariateSpline
import matplotlib.pyplot as plt
from src import parse_arguments, read_data_deeponet


class NumpySinFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        input = input.detach()
        result = np.sin(input.numpy())
        ctx.save_for_backward(input)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0].numpy()
        #np_grad_output = grad_output.numpy()
        result = np.cos(inputs)
        return grad_output.new(result)


def numpy_sin_func(input):
    return NumpySinFunction.apply(input)


def test_numpy_sin_layer():
    inputs = torch.tensor([0, np.pi/3, np.pi/2, np.pi/4], requires_grad=True)
    result = numpy_sin_func(inputs)
    print(result ** 2)
    result.backward(torch.ones_like(result))
    print(inputs.grad ** 2)


def test_Rbf_interpolator():
    N = 64
    xi, yi = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    xi = xi.reshape(-1, )
    yi = yi.reshape(-1, )

    def func(x, y):
        return np.exp(x +(y+0.25) ** 2)

    di = func(xi, yi)
    rbfi = Rbf(xi, yi, di)

    x2 = np.random.random(100)
    y2 = np.random.random(100)
    d2 = rbfi(x2, y2)
    d2_c = func(x2, y2)

    plt.subplot(1, 3, 1)
    plt.scatter(x2, y2, c=d2)

    plt.subplot(1, 3, 2)
    plt.scatter(x2, y2, c=d2_c)

    plt.subplot(1, 3, 3)
    plt.scatter(x2, y2, c=np.log10(np.abs(d2 - d2_c)))
    plt.colorbar()
    plt.show()


def test_Rbf_interpolator_sdf():
    args = parse_arguments()
    train_ds, val_ds = read_data_deeponet(args, n_data=100)
    img, xp, sdfp = next(iter(train_ds))
    print(3)

    N = 128
    m = 100
    x = np.linspace(-1, 1, N)
    xi, yi = np.meshgrid(x, x)
    xi = xi.reshape(-1, )
    yi = yi.reshape(-1, )
    sdfi = img.reshape(-1, )
    rbfi = Rbf(xi, yi, sdfi)
    sdfi_interp = rbfi(xp[:m, 0], xp[:m, 1])

    sdfi_interp = sdfi_interp.reshape(-1)
    sdfp = sdfp.numpy().reshape(-1)[:m]

    plt.subplot(1, 3, 1)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=sdfi_interp)
    plt.subplot(1, 3, 2)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=sdfp)
    plt.subplot(1, 3, 3)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=np.log10(np.abs(sdfi_interp - sdfp)))
    plt.colorbar()
    plt.show()


def test_rectbs_interpolator_sdf():
    args = parse_arguments()
    train_ds, val_ds = read_data_deeponet(args, n_data=100)
    img, xp, sdfp = next(iter(train_ds))


    N = 128
    m = 100
    x = np.linspace(-1, 1, N)
    rect_bs_interpolator = RectBivariateSpline(x, x, img.squeeze())

    sdfp_interp = np.array([rect_bs_interpolator(yri, xri).item() for (xri, yri) in xp[:m, :2]])
    sdfp = sdfp.numpy().reshape(-1)[:m]

    plt.subplot(1, 3, 1)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=sdfp_interp)

    plt.subplot(1, 3, 2)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=sdfp)

    plt.subplot(1, 3, 3)
    plt.scatter(xp[:m, 0], xp[:m, 1], c=np.log10(np.abs(sdfp - sdfp_interp)))
    plt.colorbar()
    plt.show()


def test_rectbs_interpolator():
    def func(x, y):
        return np.exp(x + (y + 0.25) ** 2)
    N = 64
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    D = func(X, Y)
    rect_bs_interpolator = RectBivariateSpline(x, x, D)

    xr = np.random.random(100)
    yr = np.random.random(100)
    dr = np.array([rect_bs_interpolator(yri, xri).item() for xri, yri in zip(xr, yr)])
    dr_c = func(xr, yr)

    plt.subplot(1, 3, 1)
    plt.scatter(xr, yr, c=dr)

    plt.subplot(1, 3, 2)
    plt.scatter(xr, yr, c=dr_c)

    plt.subplot(1, 3, 3)
    plt.scatter(xr, yr, c=np.log10(np.abs(dr - dr_c)))
    plt.colorbar()
    plt.show()

# # test_rectbs_interpolator_sdf()
# # test_rectbs_interpolator()
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import sys
#
#
# class RBF(nn.Module):
#     def __init__(self, in_features, out_features, basis_func):
#         super(RBF, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
#         self.sigmas = nn.Parameter(torch.Tensor(out_features))
#         self.basis_func = basis_func
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.normal_(self.centres, 0, 1)
#         nn.init.constant_(self.sigmas, 1)
#
#     def forward(self, input):
#         size = (input.size(0), self.out_features, self.in_features)
#         x = input.unsqueeze(1).expand(size)
#         c = self.centres.unsqueeze(0).expand(size)
#         distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
#         return self.basis_func(distances)
#
#
# class Network(nn.Module):
#     def __init__(self, layer_widths, layer_centres, basis_func):
#         super(Network, self).__init__()
#         self.rbf_layers = nn.ModuleList()
#         self.linear_layers = nn.ModuleList()
#         for i in range(len(layer_widths) - 1):
#             self.rbf_layers.append(RBF(layer_widths[i], layer_centres[i], basis_func))
#             self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i + 1]))
#
#     def forward(self, x):
#         out = x
#         for i in range(len(self.rbf_layers)):
#             out = self.rbf_layers[i](out)
#             out = self.linear_layers[i](out)
#         return out
#
#
#
# import os
# from src import (read_data, read_data_deeponet, TrainLogger, get_dtype, get_device,
#                  get_loss_func, get_save_name, get_optimizer)
# from src.network import get_network
#
# import torch
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# args = parse_arguments()
#
# dtype              = get_dtype(args)
# network_id         = args.net_id
# save_name          = get_save_name(args)
# num_epochs         = args.n_epochs
# save_every         = args.save_every
# batch_size         = args.batch_size
# loss_fn            = args.loss_fn
# checkpoint_dir     = args.ckpt_dir
# network_save_dir   = os.path.join(checkpoint_dir, 'networks')
# runs_save_dir      = os.path.join(checkpoint_dir, 'runs')
#
# train_ds, val_ds = read_data_deeponet(args, n_data=100)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size=10, shuffle=False, pin_memory=True, drop_last=True)
#
# # set cpu/gpu device
# device = get_device(args)
#
# # read network and setup optimizer, loss
# loss_fn = get_loss_func(loss_fn)
# def multiquadric(alpha):
#     phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
#     return phi
# net = Network([2, 1], [10], multiquadric).to(device=device, dtype=dtype)
# optimizer, scheduler = get_optimizer(net, args)
#
# tf_writer = SummaryWriter(os.path.join(runs_save_dir, save_name))
# train_log_writer = TrainLogger(os.path.join(runs_save_dir, save_name + "_training_logs"), optimizer)
#
# n_points_per_forward_pass = args.deeponet_npoints_per_pass
# n_points_tot = 10000
#
# x = np.linspace(-1, 1, 128)
# xi, yi = np.meshgrid(x, x)
# xi = xi.reshape(-1, 1)
# yi = yi.reshape(-1, 1)
# xyi = np.concatenate([xi, yi], axis=1)
# xyi = torch.from_numpy(xyi).to(device=device, dtype=dtype)
#
# for epoch in range(num_epochs):
#     net.train()
#     epoch_loss = 0
#     for xb, xbp, ybp in train_loader:
#         points_idx = torch.randint(0, n_points_tot, (n_points_per_forward_pass, ))
#         xb = xb.to(device=device, dtype=dtype)
#         xbp = xbp[:, points_idx, :].to(device=device, dtype=dtype)
#         ybp = ybp[:, points_idx].to(device=device, dtype=dtype)
#         optimizer.zero_grad()
#         input_data = torch.cat([xyi, xb.view(-1, 1)], dim=1)
#         pred = net(input_data)
#         loss = loss_fn(pred, ybp)
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     epoch_loss /= len(train_loader)
#     tf_writer.add_scalar("Loss/train", epoch_loss, epoch)
#
#     epoch_lossv = 0
#     net.eval()
#     for xbv, xbpv, ybpv in val_loader:
#         points_idx = torch.randint(0, n_points_tot, (n_points_per_forward_pass,))
#         xbv = xbv.to(device=device, dtype=dtype)
#         xbpv = xbpv[:, points_idx, :].to(device=device, dtype=dtype)
#         ybpv = ybpv[:, points_idx].to(device=device, dtype=dtype)
#         predv = net(xbv, xbpv)
#         lossv = loss_fn(predv, ybpv)
#         epoch_lossv += lossv.item()
#
#     epoch_lossv /= len(val_loader)
#     scheduler.step(epoch_lossv)
#
#     tf_writer.add_scalar("Loss/val", epoch_lossv, epoch)
#
#     if epoch % save_every == 0 or epoch == num_epochs - 1:
#         print(f"epoch {epoch},  ", end="")
#         print("training loss=%0.4f,  " % epoch_loss, end="")
#         print("validation loss=%0.4f,  " % epoch_lossv)
#         if not os.path.exists(checkpoint_dir):
#             os.mkdir(checkpoint_dir)
#         torch.save(net.state_dict(), os.path.join(network_save_dir, save_name + ".pth"))
#         train_log_writer.write_training_step(epoch)
# tf_writer.flush()
# tf_writer.close()
# train_log_writer.close()
from scipy.ndimage import distance_transform_edt
N = 128
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)#np.mgrid[:N, :N]
# img = np.zeros((N, N))
# img[int(N/5):int(N/3), int(N/3):int(N/2)] = 1
# img[int(N/4):int(N/2), int(N/10):int(N/5)] = 1
# img[(X-0.3)**2 + (Y+0.3)**2 < 0.3] = 1
img = np.load("../../../Downloads/img.npy").astype(int)
dist_in, indices = distance_transform_edt(img, return_indices=True)
indices_out = np.unique(indices[:, img > 0], axis=1)
boundary_out = np.array([x[indices_out[1, :]], x[indices_out[0, :]]])
dist_out, indices = distance_transform_edt(1-img, return_indices=True)
indices_in = np.unique(indices[:, img < 1], axis=1)
boundary_in = np.array([x[indices_in[1, :]], x[indices_in[0, :]]])

sdf = (dist_out - dist_in) / (N//2)

plt.contourf(x, x, sdf, 40)
plt.contour(x, x, sdf, levels=[0])
plt.scatter(boundary_in[0, :], boundary_in[1, :], c='r', marker='.')
plt.scatter(boundary_out[0, :], boundary_out[1, :], c='y', marker='.')

pnts = np.random.random((30, 2)) * 2 - 1
pnts[-1, 0] = x[10]
pnts[-1, 1] = x[100]
for p in pnts:
    argmin_in  = np.linalg.norm(boundary_in - p.reshape(2, 1), axis=0).argmin()
    argmin_out = np.linalg.norm(boundary_out - p.reshape(2, 1), axis=0).argmin()
    dist_in  = np.linalg.norm(boundary_in - p.reshape(2, 1), axis=0).min()
    dist_out = np.linalg.norm(boundary_out - p.reshape(2, 1), axis=0).min()
    sdf_p = dist_in if dist_in > dist_out else dist_out * (-1)
    print(p, sdf_p)
    plt.plot(p[0], p[1], 'r*')
    plt.plot([p[0], boundary_in[0, argmin_in]], [p[1], boundary_in[1, argmin_in]], 'k--')
    # plt.plot(boundary_out[0, argmin_out], boundary_out[1, argmin_out], marker='s')


import scipy.interpolate as si
func = si.interp2d(X, Y, sdf)
def fmt(x, y):
    z = np.take(func(x, y), 0)
    return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)


plt.gca().format_coord = fmt
plt.show()


