import torch

import vaeitemgen
from vaeitemgen.data import AmazonFashion
from vaeitemgen.loaders import DataLoaderWithImages
from vaeitemgen.models import TrainerCVAE
from vaeitemgen.nets import CFVAE_net

# get data
d = AmazonFashion("./dataset/AmazonFashion6ImgPartitioned.npy", min_n_items=10, min_n_users=10)
# create network architecture
v = CFVAE_net(d.n_users, 100).to(vaeitemgen.device)
# create loaders for training and validation
tr_loader = DataLoaderWithImages(d.folds["train"], d.item_images, d.u_i_matrix, 64, 1, 64)
val_loader = DataLoaderWithImages(d.folds["val"], d.item_images, d.u_i_matrix, 64, 100, 64, shuffle=False)
# training the model
optimizer = torch.optim.Adam(v.parameters(), lr=0.001, weight_decay=0.0001)
model = TrainerCVAE(v, optimizer)
model.validate(val_loader)
# model.train(tr_loader, val_loader)
