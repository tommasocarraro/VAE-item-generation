import vaeitemgen
from vaeitemgen.metrics import auc, ndcg_at_k, hit_at_k
import torch
import numpy as np
from tqdm import tqdm
from time import sleep
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# TODO fare anche una valutazione della generazione. Gli item generati per quel utente devono avere uno score alto -> consiglio Guglielmo
# TODO fare in modo che la loss di ricostruzione penalizzi la ricostruzione di item che non piacciono all'utente -> paper Luca
# TODO se capisco il significato delle features latenti posso metterle in 0/1 e fare delle regole logiche spegnendo o accendendo neuroni su quelle feature della condizione
# TODO devo decidere quale e' il criterio per la validazione. Non posso fare la somma dei due. Forse sarebbe meglio utilizzare la loss in validazione


class TrainerCVAE:
    """
    Trainer to train and validate the collaborative VAE model.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.mse_mean = torch.nn.MSELoss()
        self.mse_sum = torch.nn.MSELoss(reduction="sum")
        self.kl = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))

    def train(self, train_loader, val_loader, n_epochs=200, early=None, verbose=1, save_path=None):
        """
        Method for the training of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        """
        best_val_score = 0.0
        early_counter = 0

        for epoch in range(n_epochs):
            # training step
            train_loss, rating_loss, kl_loss, rec_loss = self.train_epoch(train_loader, epoch + 1)
            # validation step
            auc_score, ndcg_score, hit_score, mse_score = self.validate(val_loader)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Rating loss %.3f - KL loss %.3f - Rec loss %.3f - "
                      "Validation AUC %.3f - Validation NDCG@10 %.3f - Validation HIT@10 %.3f - Validation MSE %.3f"
                      % (epoch + 1, train_loss, rating_loss, kl_loss, rec_loss, auc_score, ndcg_score, hit_score, mse_score))
            # save best model and update early stop counter, if necessary
            if auc_score > best_val_score:
                best_val_score = auc_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader, epoch):
        """
        Method for the training of one single epoch.
        :param train_loader: data loader for training data
        :param epoch: epoch index just for printing information about training
        :return: training loss value averaged across training batches
        """
        train_loss, rating_loss, kl_loss, rec_loss = 0.0, 0.0, 0.0, 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for u_idx, item_images in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                self.optimizer.zero_grad()
                rating_pred, mu, log_var, rec_images = self.model(u_idx, item_images[:, 0])
                n_rating_pred, n_mu, n_log_var, n_rec_images = self.model(u_idx, item_images[:, 1])
                rating_loss = - torch.mean(torch.log(torch.nn.Sigmoid()(rating_pred - n_rating_pred)))
                rec_loss = (self.mse_sum(item_images[:, 0], rec_images) +
                            self.mse_sum(item_images[:, 1], n_rec_images)) / 2
                kl_loss = (self.kl(mu, log_var) + self.kl(n_mu, n_log_var)) / 2
                loss = rating_loss + rec_loss + kl_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix({"Loss": loss.item(), "Rating loss": rating_loss.item(), "KL loss": kl_loss.item(),
                                    "Rec loss": rec_loss.item()})
                sleep(0.1)
        return train_loss / len(train_loader), rating_loss / len(train_loader), \
               kl_loss / len(train_loader), rec_loss / len(train_loader)

    def predict(self, u_idx, item_image):
        """
        Method for performing a prediction of the model. It returns the score for the given user and item image.
        In addition, it returns the reconstruction of the given item image.

        :param u_idx: user for which the prediction has to be computed
        :param item_image: image for which the prediction has to be computed
        :return: the prediction of the model for the given image-user pair. The prediction is composed of a
        recommendation score and the reconstruction of the item image.
        """
        with torch.no_grad():
            pred_ratings, _, _, rec_images = self.model(u_idx, item_image)
            return pred_ratings, rec_images

    def validate(self, val_loader):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :return: validation AUC, NDCG@10, HIT@10 and MSE of images averaged across validation examples
        """
        auc_score, ndcg_score, hit_score, mse_score = [], [], [], []
        for batch_idx, (u_idx, item_images) in enumerate(val_loader):
            predicted_scores, rec_images = self.predict(u_idx, item_images[:, 0])
            n_predicted_scores, n_rec_images = self.predict(u_idx, item_images[:, 1:])
            predicted_scores = predicted_scores.cpu().numpy()
            n_predicted_scores = n_predicted_scores.cpu().numpy()
            auc_score.append(auc(predicted_scores, n_predicted_scores))
            mse_score.append((self.mse_mean(item_images[:, 0], rec_images) +
                              self.mse_mean(item_images[:, 1:], n_rec_images) / 2).detach().cpu().numpy())
            gt = np.zeros((u_idx.shape[0], item_images.shape[1]))
            gt[:, 0] = 1
            total_preds = np.concatenate([predicted_scores[:, np.newaxis], n_predicted_scores], axis=1)
            ndcg_score.append(np.mean(ndcg_at_k(total_preds, gt, 10)))
            hit_score.append(np.mean(hit_at_k(total_preds, gt, 10)))

        # plot images and their reconstruction

        print("Qualitative evaluation")

        print("Reconstruction quality")

        with torch.no_grad():
            rec_images = rec_images[-4:].view(-1, 3, 64, 64)
            images = torch.cat([item_images[:, 0][-4:], rec_images], dim=0)
            grid = make_grid(images, nrow=4)
            plt.figure(figsize=(15, 5))
            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)), interpolation='nearest', cmap='gray')
            plt.show()

        # plot generated images

        print("Generation quality")

        with torch.no_grad():
            eps = torch.randn((10, 100)).to(vaeitemgen.device)
            u = torch.tensor([3 for _ in range(10)]).to(vaeitemgen.device)
            gen_images = self.model.decode(eps, u).view(-1, 3, 64, 64)
            grid = make_grid(gen_images, nrow=2)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)), interpolation='nearest', cmap='gray')
            plt.show()

        return np.mean(auc_score), np.mean(ndcg_score), np.mean(hit_score), np.mean(mse_score)

    def save_model(self, path):
        """
        Method for saving the model.
        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.
        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TrainerMF:
    """
    Trainer to train and validate the Matrix Factorization model.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, n_epochs=200, early=None, verbose=1, save_path=None):
        """
        Method for the training of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        """
        best_val_score = 0.0
        early_counter = 0

        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader, epoch + 1)
            # validation step
            auc_score, ndcg_score, hit_score = self.validate(val_loader)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Validation AUC %.3f - Validation NDCG@10 %.3f - "
                      "Validation HIT@10 %.3f"
                      % (epoch + 1, train_loss, auc_score, ndcg_score, hit_score))
            # save best model and update early stop counter, if necessary

            if auc_score > best_val_score:
                best_val_score = auc_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader, epoch):
        """
        Method for the training of one single epoch.
        :param train_loader: data loader for training data
        :param epoch: epoch index just for printing information about training
        :return: training loss value averaged across training batches
        """
        train_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for u_idx, i_idx in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                self.optimizer.zero_grad()
                rating_pred = self.model(u_idx, i_idx[:, 0])
                n_rating_pred = self.model(u_idx, i_idx[:, 1])
                loss = - torch.mean(torch.log(torch.nn.Sigmoid()(rating_pred - n_rating_pred)))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix({"Loss": loss.item()})
                sleep(0.1)
        return train_loss / len(train_loader)

    def predict(self, u_idx, i_idx):
        """
        Method for performing a prediction of the model. It returns the score for the given user and item index.

        :param u_idx: user for which the prediction has to be computed
        :param i_idx: item for which the prediction has to be computed
        :return: the prediction of the model for the given user_item pair.
        """
        with torch.no_grad():
            pred_ratings = self.model(u_idx, i_idx)
            return pred_ratings

    def validate(self, val_loader):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :return: validation AUC and MSE averaged across validation examples
        """
        auc_score, ndcg_score, hit_score = [], [], []
        for batch_idx, (u_idx, i_idx) in enumerate(val_loader):
            predicted_scores = self.predict(u_idx, i_idx[:, 0]).cpu().numpy()
            n_predicted_scores = self.predict(u_idx.unsqueeze(1).expand(-1, i_idx.shape[1] - 1), i_idx[:, 1:]).cpu().\
                numpy()
            auc_score.append(auc(predicted_scores, n_predicted_scores))
            gt = np.zeros((u_idx.shape[0], i_idx.shape[1]))
            gt[:, 0] = 1
            total_preds = np.concatenate([predicted_scores[:, np.newaxis], n_predicted_scores], axis=1)
            ndcg_score.append(np.mean(ndcg_at_k(total_preds, gt, 10)))
            hit_score.append(np.mean(hit_at_k(total_preds, gt, 10)))

        return np.mean(auc_score), np.mean(ndcg_score), np.mean(hit_score)

    def save_model(self, path):
        """
        Method for saving the model.
        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.
        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
