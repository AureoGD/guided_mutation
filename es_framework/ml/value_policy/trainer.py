import torch
import torch.nn as nn
import torch.optim as optim


class ValueTrainer:

    def __init__(self, model, lr=1e-3):

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):

        self.model.train()
        total_loss = 0

        for X, y in dataloader:

            X = X.to(self.model.device)
            y = y.to(self.model.device)

            pred = self.model(X)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    def evaluate(self, dataloader):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for X, y in dataloader:

                X = X.to(self.model.device)
                y = y.to(self.model.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)

                total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)