import torch


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, latent_size):
        super(MatrixFactorization, self).__init__()
        # user embeddings randomly initialized
        self.u_emb = torch.nn.Embedding(n_users, latent_size)
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        # item embeddings randomly initialized
        self.i_emb = torch.nn.Embedding(n_items, latent_size)
        torch.nn.init.xavier_normal_(self.i_emb.weight)
        self.latent_size = latent_size

    def forward(self, u_idx, i_idx):
        return torch.sum(self.u_emb(u_idx) * self.i_emb(i_idx), dim=1)


class CFVAE_net(torch.nn.Module):
    def __init__(self, n_users, latent_size):
        super(CFVAE_net, self).__init__()
        # user embeddings randomly initialized
        self.u_emb = torch.nn.Embedding(n_users, latent_size)
        torch.nn.init.xavier_normal_(self.u_emb.weight)
        self.latent_size = latent_size

        # encoder architecture

        self.enc_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(3, 32, (3, 3), (2, 2), (1, 1)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)),
                torch.nn.ReLU(),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(128 * 8 * 8, latent_size * 2)
            ]
        )

        # decoder architecture

        self.dec_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(latent_size * 2, 128 * 8 * 8),
                torch.nn.ReLU(),
                torch.nn.Unflatten(1, (128, 8, 8)),
                torch.nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), (1, 1), (1, 1)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 32, (3, 3), (2, 2), (1, 1), (1, 1)),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 3, (3, 3), (2, 2), (1, 1), (1, 1))
            ]
        )

    def encode(self, x):
        """
        It takes input x and encodes it returning the mu and log_var in the latent space of the vae.
        The input x is the image of a product.

        :param x: image of a product of the Amazon fashion dataset
        :return: mu and log_var of the gaussian distribution for the given image
        """
        for layer in self.enc_layers:
            x = layer(x)
        mu = x[:, :self.latent_size]
        log_var = x[:, self.latent_size:]
        return mu, log_var

    def decode(self, z, u_idx):
        """
        It generates an image of a product given a latent vector and a user index in input. The user index is used
        for conditioning the generation based on the user embedding at index "u_idx".

        :param z: latent vector for generating an image
        :param u_idx: index of user for conditioning the generation
        :return: generated image through the decoder network
        """
        z = torch.hstack([z, self.u_emb(u_idx)])
        for layer in self.dec_layers:
            z = layer(z)
        return torch.sigmoid(z)

    def forward(self, u_idx, i_image):
        """
        It performs an entire forward phase of the collaborative VAE model.

        The input is an image of a product and a user index. The image is given to the encoder which produces the
        mu and log var of a Gaussian distribution. The mu is used as item embedding. The user index is used to take the
        user embedding from the embedding look-up table. Then, the rating for user and item is computed as the dot
        product of the user embedding with mu. Eventually, the decoder takes z sampled from N(mu, log_var) and the user
        embedding and tries to reconstruct the given image.

        :param u_idx: index of user
        :param i_image: image of a product
        :return: prediction for u-i rating, mu, log_var, reconstructed image
        """
        mu, log_var = self.encode(i_image.reshape(-1, 3, i_image.shape[-2], i_image.shape[-1])
                                  if len(i_image.shape) > 4 else i_image)
        # rep trick
        eps = torch.randn_like(mu)
        std = torch.exp(log_var * 0.5)
        z = (std * eps) + mu
        rec = self.decode(z, u_idx.repeat(i_image.shape[1]) if len(i_image.shape) > 4 else u_idx)
        if len(i_image.shape) > 4:
            rec = rec.view(u_idx.shape[0], i_image.shape[1], 3, i_image.shape[-2], i_image.shape[-1])
        # get prediction for rating of given user and item
        pred = torch.sum((self.u_emb(u_idx).repeat(i_image.shape[1], 1)
                          if len(i_image.shape) > 4 else self.u_emb(u_idx)) * mu, dim=1)
        if len(i_image.shape) > 4:
            pred = pred.view(u_idx.shape[0], i_image.shape[1])
        return pred, mu, log_var, rec
