import torch
import os
from model import *
from dataloader import *
import torch.nn.functional as F
import numpy as np


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, epochs, optimizer, loss_fn, train_loader):
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:

                # forward pass
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels.squeeze())
                running_loss += loss.item()

                # backward pass
                optimizer.zero_grad()  # Clear gradients w.r.t. parameters
                loss.backward()  # Getting gradients w.r.t. parameters
                optimizer.step()  # Updating parameters

            print(f"epoch {epoch+1}, loss {running_loss}")

    def infer(self, test_loader, loss_fn):
        all_predictions = []
        loss = 0
        match_count = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                predictions_softmax = self.model(inputs)
                loss += loss_fn(predictions_softmax, labels.squeeze())
                predictions_class = torch.max(predictions_softmax, 1)
                match_count += np.sum(
                    predictions_class.indices.numpy() == labels.squeeze().numpy()
                )
                all_predictions.extend(predictions_class.indices.numpy().tolist())
        batch_size = 100
        return match_count / (len(test_loader) * batch_size), loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "model.pkl"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pkl")))
