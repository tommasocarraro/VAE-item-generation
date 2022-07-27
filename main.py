import torch

import vaeitemgen
from vaeitemgen.data import AmazonFashion
from vaeitemgen.loaders import DataLoader
from vaeitemgen.models import Trainer
from vaeitemgen.nets import CFVAE_net

# get data
d = AmazonFashion("./dataset/AmazonFashion6ImgPartitioned.npy", min_n_items=3, min_n_users=3)
# create network architecture
v = CFVAE_net(d.n_users, 100).to(vaeitemgen.device)
# create loaders for training and validation
tr_loader = DataLoader(d.folds["train"], d.item_images, d.u_i_matrix, 64, 32)
val_loader = DataLoader(d.folds["val"], d.item_images, d.u_i_matrix, 64, 32, shuffle=False)
# training the model
optimizer = torch.optim.Adam(v.parameters(), lr=0.001, weight_decay=0.0001)
model = Trainer(v, optimizer)
model.train(tr_loader, val_loader)
