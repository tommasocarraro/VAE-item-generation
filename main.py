import torch

import vaeitemgen
from vaeitemgen.data import AmazonFashion
from vaeitemgen.loaders import DataLoader
from vaeitemgen.models import Trainer
from vaeitemgen.nets import CFVAE_net

# get data
d = AmazonFashion("./dataset/AmazonFashion6ImgPartitioned.npy")
# create network architecture
v = CFVAE_net(d.n_users, 100).to(vaeitemgen.device)
# create loaders for training and validation
tr_loader = DataLoader(d.folds["train"], d.item_images, d.u_i_matrix, 224, 64)
val_loader = DataLoader(d.folds["val"], d.item_images, d.u_i_matrix, 224, 64, shuffle=False)
# training the model
optimizer = torch.optim.Adam(v.parameters(), lr=0.001)
model = Trainer(v, optimizer)
model.train(tr_loader, val_loader)

# import math
#
# def compute_out_shape_conv(h_in, w_in, k, p=(0, 0), d=(1, 1), s=(1, 1)):
#     h_out = math.floor((h_in + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
#     w_out = math.floor((w_in + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
#     return h_out, w_out
#
# def compute_out_shape_deconv(h_in, w_in, k, p=(0, 0), d=(1, 1), s=(1, 1), o_p=(0, 0)):
#     h_out = (h_in - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + o_p[0] + 1
#     w_out = (w_in - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + o_p[1] + 1
#     return h_out, w_out
#
# h_out, w_out = compute_out_shape_conv(224, 224, (11, 11), s=(4, 4))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (2, 2), s=(2, 2))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (5, 5), s=(1, 1), p=(2, 2))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (2, 2), s=(2, 2))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (3, 3), s=(1, 1), p=(1, 1))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (3, 3), s=(1, 1), p=(1, 1))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (3, 3), s=(1, 1), p=(1, 1))
# h_out, w_out = compute_out_shape_conv(h_out, w_out, (2, 2), s=(2, 2))
#
# h_out, w_out = math.floor(h_out * 2), math.floor(w_out * 2)
# h_out, w_out = compute_out_shape_deconv(h_out, w_out, (3, 3), p=(2, 2), s=(1, 1), d=(2, 2), o_p=(1, 1))
# h_out, w_out = compute_out_shape_deconv(h_out, w_out, (3, 3), p=(1, 1), s=(1, 1))
# h_out, w_out = compute_out_shape_deconv(h_out, w_out, (3, 3), p=(1, 1), s=(1, 1))
# h_out, w_out = math.floor(h_out * 2), math.floor(w_out * 2)
# h_out, w_out = compute_out_shape_deconv(h_out, w_out, (5, 5), p=(4, 4), s=(1, 1), d=(2, 2), o_p=(1, 1))
# h_out, w_out = math.floor(h_out * 2), math.floor(w_out * 2)
# h_out, w_out = compute_out_shape_deconv(h_out, w_out, (11, 11), s=(4, 4), o_p=(1, 1))
# print()
