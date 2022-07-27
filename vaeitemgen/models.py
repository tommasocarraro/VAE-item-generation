from vaeitemgen.metrics import auc
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


def kl_loss_function(mu, log_var): return -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - torch.square(mu))


class Trainer:
    """
    Trainer to train and validate the collaborative VAE model.
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
            train_loss, rating_loss, kl_loss, rec_loss = self.train_epoch(train_loader, epoch + 1)
            # validation step
            auc_score, mse_score, val_loss = self.validate(val_loader)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Rating loss %.3f - KL loss %.3f - Rec loss %.3f - "
                      "Validation AUC %.3f - Validation MSE %.3f"
                      % (epoch + 1, train_loss, rating_loss, kl_loss, rec_loss, auc_score, mse_score))
            # save best model and update early stop counter, if necessary
            val_score = val_loss
            if val_score > best_val_score:
                best_val_score = val_score
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
                rating_pred, mu, log_var, rec_images = self.model(item_images[:, 0], u_idx)
                n_rating_pred, n_mu, n_log_var, n_rec_images = self.model(item_images[:, 1], u_idx)
                rating_loss = - torch.sum(torch.log(torch.nn.Sigmoid()(rating_pred - n_rating_pred)))
                rec_loss = (torch.nn.MSELoss(reduction="sum")(item_images[:, 0], rec_images) +
                            torch.nn.MSELoss(reduction="sum")(item_images[:, 1], n_rec_images)) / 2
                kl_loss = (kl_loss_function(mu, log_var) + kl_loss_function(n_mu, n_log_var)) / 2
                loss = rating_loss + rec_loss + kl_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix({"Loss": loss.item(), "Rating loss": rating_loss.item(), "KL loss": kl_loss.item(),
                                    "Rec loss": rec_loss.item()})
                sleep(0.1)
        return train_loss / len(train_loader), rating_loss / len(train_loader), \
               kl_loss / len(train_loader), rec_loss / len(train_loader)

    def predict(self, item_image, u_idx):
        """
        Method for performing a prediction of the model. It returns the score for the given image and user index.
        In addition, it returns the reconstruction of the given item image.

        :param item_image: image for which the prediction has to be computed
        :param u_idx: user for which the prediction has to be computed
        :return: the prediction of the model for the given image-user pair. The prediction is composed of a
        recommendation score and the reconstruction of the item image.
        """
        with torch.no_grad():
            pred_ratings, _, _, rec_images = self.model(item_image, u_idx)
            return pred_ratings, rec_images

    def validate(self, val_loader):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :return: validation AUC and MSE averaged across validation examples
        """
        auc_score, mse_score, val_loss = [], [], 0.0
        for batch_idx, (u_idx, item_images) in enumerate(val_loader):
            predicted_scores, rec_images = self.predict(item_images[:, 0], u_idx)
            n_predicted_scores, n_rec_images = self.predict(item_images[:, 1], u_idx)
            auc_score.append(auc(predicted_scores.detach().cpu().numpy(), n_predicted_scores.detach().cpu().numpy()))
            mse_score.append((torch.nn.MSELoss()(item_images[:, 0], rec_images) +
                              torch.nn.MSELoss()(item_images[:, 1], n_rec_images)) / 2)

            # compute validation loss

            rating_loss = - torch.mean(torch.log(torch.nn.Sigmoid()(predicted_scores - n_predicted_scores)))
            rec_loss = (torch.nn.MSELoss(reduction="sum")(item_images[:, 0], rec_images) +
                        torch.nn.MSELoss(reduction="sum")(item_images[:, 1], n_rec_images)) / 2
            val_loss += (rating_loss + rec_loss)

        # plot images and their reconstruction

        with torch.no_grad():
            rec_images = rec_images[-4:].view(-1, 3, 64, 64)
            images = torch.cat([item_images[:, 0][-4:], rec_images], dim=0)
            grid = make_grid(images, nrow=4)
            plt.figure(figsize=(15, 5))
            plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest', cmap='gray')
            plt.show()

        # plot generated images

        with torch.no_grad():
            eps = torch.randn((10, 100))
            u = torch.tensor([3 for _ in range(10)])
            gen_images = self.model.decode(eps, u).view(-1, 3, 64, 64)
            grid = make_grid(gen_images, nrow=2)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest', cmap='gray')
            plt.show()

        return np.mean(auc_score), np.mean(mse_score), val_loss / len(val_loader)

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
