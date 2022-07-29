import torch
import vaeitemgen
from vaeitemgen.data import AmazonFashion
from vaeitemgen.loaders import DataLoader
from vaeitemgen.models import TrainerMF
from vaeitemgen.nets import MatrixFactorization

# get data
d = AmazonFashion("./dataset/AmazonFashion6ImgPartitioned.npy", with_images=False)
# create network architecture
v = MatrixFactorization(d.n_users, d.n_items, 100).to(vaeitemgen.device)
# create loaders for training and validation
tr_loader = DataLoader(d.folds["train"], d.u_i_matrix, 64)
val_loader = DataLoader(d.folds["val"], d.u_i_matrix, 64, n_neg=100, shuffle=False)
# training the model
optimizer = torch.optim.Adam(v.parameters(), lr=0.001, weight_decay=0.0001)
model = TrainerMF(v, optimizer)
model.train(tr_loader, val_loader)
