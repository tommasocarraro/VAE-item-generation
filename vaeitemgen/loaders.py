import numpy as np
import torch
from PIL import Image
import vaeitemgen


class DataLoaderWithImages:
    """
    Data loader to load the training/validation/test set of the Amazon fashion dataset. It creates batches for
    pair-wise learning procedure and evaluation. Each batch is composed of:
    - user_idx: indexes of the users. These indexes are used to take the user embeddings from an embedding look-up
    table;
    - item_images: images of the items. These are used to compute the embeddings for the items by using a CNN encoder.
    Note that item_images is an array where the first column contains the positive items, while the second one the
    negative items. This structure is used for the pair-wise learning procedure.

    During training, the batch is used to perform the pair-wise learning procedure (also referred as BPR procedure).
    During evaluation, the batch is used to get the prediction for positive and negative interactions. Then, the AUC
    measure is computed.
    """
    def __init__(self,
                 data,
                 item_images,
                 u_i_matrix,
                 image_size,
                 batch_size=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: np.array of triples (user, item, rating)
        :param item_images: list of images of the products of the dataset. Image at index 0 is the image of product with
        index 0 in the dataset
        :param u_i_matrix: sparse user-item interaction matrix where there is a 1 if user i interacted with item j
        :param image_size: size of images of products. Images will be resized to the given size. Default is 224x224
        as they used in the paper about fashion recommendation generation
        :param batch_size: batch size for the training of the model
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = data
        self.item_images = item_images
        self.u_i_matrix = u_i_matrix
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            positive_item_images = [np.asarray(Image.open(image_bytes).convert('RGB').
                                               resize((self.image_size, self.image_size))).transpose((2, 0, 1))
                                    for image_bytes in self.item_images[data[:, 1]]]
            u_idx = data[:, 0]

            # get negative item images

            user_interactions = self.u_i_matrix[u_idx]
            negative_item_images = [np.asarray(Image.open(
                self.item_images[np.random.permutation(
                    (1 - user_interactions[u].todense()).nonzero()[1])[0]]).convert('RGB').
                                                resize((self.image_size, self.image_size))
                                                ).transpose((2, 0, 1))
                                     for u in range(user_interactions.shape[0])]

            yield torch.tensor(u_idx).to(vaeitemgen.device), \
                  torch.tensor(np.stack([np.stack(positive_item_images),
                                         np.stack(negative_item_images)], axis=1)).float().to(vaeitemgen.device) / 255


class DataLoader:
    """
    Same data loader as DataLoaderWithImages where instead of retrieving images for items it retrieves item indexes for
    learning simple ML recommendation models like MF.
    """
    def __init__(self,
                 data,
                 u_i_matrix,
                 batch_size=1,
                 n_neg=1,
                 shuffle=True):
        """
        Constructor of the training data loader.
        :param data: np.array of triples (user, item, rating)
        :param u_i_matrix: sparse user-item interaction matrix where there is a 1 if user i interacted with item j
        :param batch_size: batch size for the training of the model
        :param n_neg: number of negative items that have to be sampled and given with the positive item. 1 during
        training and 100 during validation and test
        :param shuffle: whether to shuffle data during training or not
        """
        self.data = data
        self.u_i_matrix = u_i_matrix
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            u_idx = data[:, 0]
            p_i_idx = data[:, 1]

            # get negative item indexes

            user_interactions = self.u_i_matrix[u_idx]
            n_i_idx = np.stack([np.random.permutation((1 - user_interactions[u].todense()).nonzero()[1])[:self.n_neg]
                                for u in range(user_interactions.shape[0])])

            yield torch.tensor(u_idx).to(vaeitemgen.device), \
                  torch.tensor(np.concatenate([p_i_idx[:, np.newaxis], n_i_idx], axis=1)).to(vaeitemgen.device)
