import os
import numpy as np

import torch
import torch.nn as nn


class Trainer():
    def __init__(
        self,
        model,
        loss,
        optimizer,
        metrics=[],
        checkpoint="../model/model.pth.tar",
        initial=False
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint

        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.metrics = metrics

        if initial:
            self.remove_checkpoint()
        else:
            self.load_checkpoint()

    def save_checkpoint(self):
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            self.checkpoint
        )

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint):
            checkpoint = torch.load(self.checkpoint)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def remove_checkpoint(self):
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def predict(self, loader):
        self.model.eval()

        results = []
        with torch.no_grad():
            for haze, image in loader:
                haze = haze.to(self.device)
                pred = self.model(haze).clamp_(0, 1)
                results.append((image, haze, pred))
        self.model.train()

        return results

    def accuracy(self, loader):
        if loader is None:
            return None

        result = dict()
        for metric in self.metrics:
            result[metric] = []

        self.model.eval()
        with torch.no_grad():
            for haze, image in loader:
                haze = haze.to(self.device)
                pred = self.model(haze).cpu()

                for metric in self.metrics:
                    result[metric].append(self.metrics[metric](pred, image))
        self.model.train()

        for metric in result:
            result[metric] = np.mean(result[metric])

        return result

    def train(self, loader):
        pass

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=1
    ):
        self.load_checkpoint()
        self.accuracy(val_loader)

        historial = []
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            loss = self.train(train_loader)
            accuracy = self.accuracy(val_loader)
            self.save_checkpoint()

            historial.append((loss, accuracy))
        return historial


class TrainerGan():
    def __init__(
        self,
        model_D,
        model_G,
        optimizer_D,
        optimizer_G,
        initial=False,
        l1_lambda=100,
        checkpoint_D="../model/pix2pix_disc.pth.tar",
        checkpoint_G="../model/pix2pix_gen.pth.tar",
        results='../images/pix2pix/'
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.l1_lambda = l1_lambda
        self.results = results
        self.checkpoint_D = checkpoint_D
        self.checkpoint_G = checkpoint_G

        self.model_D = model_D.to(self.device)
        self.model_G = model_G.to(self.device)
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

        self.loss_BCE = nn.BCEWithLogitsLoss()
        self.loss_L1 = nn.L1Loss()

        if initial:
            self.remove_checkpoint()

    def save_checkpoint(self):
        torch.save(
            {
                'state_dict': self.model_D.state_dict(),
                'optimizer': self.optimizer_D.state_dict()
            },
            self.checkpoint_D
        )

        torch.save(
            {
                'state_dict': self.model_G.state_dict(),
                'optimizer': self.optimizer_G.state_dict()
            },
            self.checkpoint_G
        )

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_D):
            checkpoint_D = torch.load(self.checkpoint_D)
            self.model_D.load_state_dict(checkpoint_D["state_dict"])
            self.optimizer_D.load_state_dict(checkpoint_D["optimizer"])

        if os.path.exists(self.checkpoint_G):
            checkpoint_G = torch.load(self.checkpoint_G)
            self.model_G.load_state_dict(checkpoint_G["state_dict"])
            self.optimizer_G.load_state_dict(checkpoint_G["optimizer"])

    def remove_checkpoint(self):
        if os.path.exists(self.checkpoint_D):
            os.remove(self.checkpoint_D)

        if os.path.exists(self.checkpoint_G):
            os.remove(self.checkpoint_G)

    def accuracy(self, loader):
        pass

    def save_images(self, loader, epoch):
        pass

    def predict(self, loader):
        pass

    def train(self, loader, scaler_D, scaler_G):
        pass

    def fit(self, train_loader, val_loader, epochs=1):
        self.load_checkpoint()
        self.accuracy(val_loader)

        scaler_D = torch.cuda.amp.GradScaler()
        scaler_G = torch.cuda.amp.GradScaler()

        historial = []
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            loss = self.train(train_loader, scaler_D, scaler_G)
            accuracy = self.accuracy(val_loader)
            self.save_images(val_loader, epoch)
            self.save_checkpoint()

            historial.append((loss, accuracy))

        return historial
