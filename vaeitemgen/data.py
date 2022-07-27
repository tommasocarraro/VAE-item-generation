import numpy as np
import io
from scipy.sparse import csr_matrix


class AmazonFashion:
    def __init__(self, data_path, min_n_items=1, min_n_users=1):
        """
        It constructs the Amazon fashion dataset.

        From the original dataset, it removes the users that have not rated at least min_n_items items and the items
        that have not been rated by at least min_n_users users.

        :param data_path: path where to find the dataset raw files
        :param min_n_items: minimum number of items that a user has to be rated to be included in the dataset.
        Default to 1.
        :param min_n_users: minimum number of ratings that an item has to be received by different users to be included
        in the dataset. Default to 1.
        """
        self.data = self.get_data(data_path)
        self.u_i_matrix, self.folds, self.item_images, self.n_users, self.n_items = self.process_data()
        if min_n_items > 1 or min_n_users > 1:
            self.reduce_sparsity(min_n_items, min_n_users)

    @staticmethod
    def get_data(path):
        """
        It takes the Amazon fashion dataset from the given path and returns a numpy array containing the folds,
        item content data, and dataset information.

        :param path: path where the dataset is stored
        """
        def convert(value):
            """
            It converts bytes to ASCII characters. This is useful for the images of the datasets. They are stored in
            the form of bytes. We convert them in ASCII and then use the ASCII to load the image.

            :param value: data structure containing the images of the dataset in the form of bytes
            :return: the same data structure, where the images are converted to ASCII characters
            """
            if isinstance(value, bytes):
                try:
                    return value.decode('ascii')
                except:
                    return value
            if isinstance(value, dict):
                return dict(map(convert, value.items()))
            if isinstance(value, tuple):
                return map(convert, value)
            return value

        data = np.load(path, allow_pickle=True, encoding='bytes')
        data[3] = convert(data[3])
        return data

    def process_data(self):
        """
        It processes the dataset and creates the train, validation, and test folds. It also creates a list to retrieve
        product images given their indexes. Index 0 in the list corresponds to the image of product with index 0. The
        image is memorized as bytes, so it has to be converted to numpy to be used.

        In addition, this function removes useless information from the dataset and creates unique integer identifiers
        for users and items.

        Eventually, this function creates the sparse user-item interaction matrix, with users on the rows and items on
        the columns. A 1 in the (i,j) position of the matrix means that item j is relevant for user i.
        """
        # create train, validation, and test folds
        folds = {}
        fold_names = ["train", "val", "test"]
        user_id, item_id = 0, 0
        user_map, item_map = {}, {}
        rows, cols = [], []  # rows and columns for creating sparse user-item matrix
        for i in range(3):
            folds[fold_names[i]] = []
            for user_idx in self.data[i]:
                for user_data in self.data[i][user_idx]:
                    if user_data["reviewerID"] not in user_map:
                        user_map[user_data["reviewerID"]] = user_id
                        user_id += 1
                    if user_data["asin"] not in item_map:
                        item_map[user_data["asin"]] = item_id
                        item_id += 1
                    folds[fold_names[i]].append((user_map[user_data["reviewerID"]], item_map[user_data["asin"]],
                                                 int(user_data["overall"])))
                    rows.append(user_map[user_data["reviewerID"]])
                    cols.append(item_map[user_data["asin"]])
            folds[fold_names[i]] = np.array(folds[fold_names[i]])

        # create user-item sparse matrix
        values = np.ones(len(rows))
        u_i_matrix = csr_matrix((values, (rows, cols)), shape=(user_id, item_id))

        # create array to retrieve product images
        item_to_image = {}
        for product_idx in self.data[3]:
            item_to_image[self.data[3][product_idx]["asin"]] = io.BytesIO(self.data[3][product_idx]["imgs"])

        item_images = []
        id_to_string_id = {v: k for k, v in item_map.items()}
        for item in range(item_id):
            item_images.append(item_to_image[id_to_string_id[item]])

        return u_i_matrix, folds, np.array(item_images), user_id, item_id

    def reduce_sparsity(self, min_n_items, min_n_users):
        # get indexes of users and items that satisfy the given conditions
        users_to_keep = (self.u_i_matrix.sum(axis=1) > min_n_items).nonzero()[0]
        items_to_keep = (self.u_i_matrix.sum(axis=0) > min_n_users).nonzero()[1]

        # filter the three folds based on the information

        for fold in self.folds:
            self.folds[fold] = self.folds[fold][np.logical_and(np.isin(self.folds[fold][:, 0], users_to_keep),
                                                               np.isin(self.folds[fold][:, 1], items_to_keep)), :]

        # after the removal of some users and items from the dataset, the indexes are not progressive anymore
        # we have some holes in the index space. We need to re-map the indexes from 0 to #users and #items.

        user_id, item_id = 0, 0
        user_map, item_map = {}, {}
        rows, cols = [], []

        new_folds = {fold: [] for fold in self.folds}

        for fold in self.folds:
            for u, i, r in self.folds[fold]:
                if u not in user_map:
                    user_map[u] = user_id
                    user_id += 1
                if i not in item_map:
                    item_map[i] = item_id
                    item_id += 1
                new_folds[fold].append((user_map[u], item_map[i], r))
                rows.append(user_map[u])
                cols.append(item_map[i])
            new_folds[fold] = np.array(new_folds[fold])

        self.folds = new_folds

        self.u_i_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(user_id, item_id))

        self.n_users = user_id
        self.n_items = item_id

        # the same consideration holds also for the array containing the images

        item_images = []
        reversed_item_map = {v: k for k, v in item_map.items()}

        for i in range(item_id):
            item_images.append(self.item_images[reversed_item_map[i]])

        self.item_images = np.array(item_images)
