import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from .base_model import Base_Model
from .losses import rmspe

from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(
        self,
        hyperparams
    ):
        super().__init__()
        self.hyperparams = hyperparams
        self._init_hyperparams()

        self.embs = nn.ModuleList([nn.Embedding(x, self.emb_dim) for x in self.n_categories])
        self.id_dim = self.emb_dim * len(self.n_categories)
        self.dropout_all = self.dropout_id + self.dropout_num
        self.dropout_id = nn.Dropout(self.dropout_id)
        if self.bn:
            self.sequence = nn.Sequential(
                nn.Linear(self.src_num_dim + self.id_dim, self.hidden),
                nn.Dropout(self.dropout_all),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.Dropout(self.dropout_all),
                nn.BatchNorm1d(self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1)
            )
        else:
            self.sequence = nn.Sequential(
                nn.Linear(self.src_num_dim + self.id_dim, self.hidden),
                nn.Dropout(self.dropout_all),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
                nn.Dropout(self.dropout_all),
                nn.ReLU(),
                nn.Linear(self.hidden, 1)
            )

    def _init_hyperparams(self):
        self.src_num_dim = self.hyperparams["src_num_dim"]
        self.n_categories = self.hyperparams["n_categories"]
        self.emb_dim = self.hyperparams["emb_dim"]
        self.dropout_id = self.hyperparams["dropout_id"]
        self.hidden = self.hyperparams["hidden"]
        self.dropout_num = self.hyperparams["dropout_num"]
        self.bn = self.hyperparams["bn"]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_num, x_id):
        embs = [embedding(x_id[:, i]) for i, embedding in enumerate(self.embs)]
        x_id_emb = self.dropout_id(torch.cat(embs, 1))
        x_all = torch.cat([x_num, x_id_emb], 1)
        x = self.sequence(x_all)
        return torch.squeeze(x)


class MLP_Dataset(Dataset):
    def __init__(self, x_num, x_id, y):
        super().__init__()
        self.x_num = x_num
        self.x_id = x_id
        self.y = y

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], torch.LongTensor(self.x_id[idx])
        else:
            return self.x_num[idx], torch.LongTensor(self.x_id[idx]), self.y[idx]


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

class MLP_Model(Base_Model):
    def __init__(
        self,
        train_data,
        valid_data,
        test_data,
        hyperparams_path
    ):
        super(MLP_Model, self).__init__(
            train_data,
            valid_data,
            test_data,
            hyperparams_path
        )
        self._generate_feature_and_target()
        self.hyperparams["model"]["src_num_dim"] = self.x_train_num.shape[1]
        self.model = MLP(self.hyperparams["model"])
        self._init_hyperparams()

    def _init_hyperparams(self):
        self.epochs = self.hyperparams["trainer"]["epochs"]
        self.lr = self.hyperparams["trainer"]["lr"]
        self.batch_size = self.hyperparams["trainer"]["batch_size"]
        self.device = self.hyperparams["trainer"]["device"]

    def _generate_feature_and_target(self):
        id_cols = ["stock_id", "time_id"]
        num_cols = [c for c in self.train_data.columns if c not in ["time_id", "stock_id", "row_id", "target"]]

        self.x_train_num = np.nan_to_num(self.train_data[num_cols].values).astype("float32")
        # self.x_train_num = self.train_data[num_cols].values.astype("float32")
        self.x_train_id = np.nan_to_num(self.train_data[id_cols].values)
        self.y_train = self.train_data["target"].values.astype("float32")

        self.x_valid_num = np.nan_to_num(self.valid_data[num_cols].values).astype("float32")
        # self.x_valid_num = self.valid_data[num_cols].values.astype("float32")
        self.x_valid_id = np.nan_to_num(self.valid_data[id_cols].values)
        self.y_valid = self.valid_data["target"].values.astype("float32")

        # np.savetxt("x_valid_num.csv", self.x_valid_num, delimiter = ',')
        # np.savetxt("x_valid_id.csv", self.x_valid_id, delimiter = ',')
        # np.savetxt("y_valid.csv", self.y_valid, delimiter = ',')

        self.x_test_num = np.nan_to_num(self.test_data[num_cols].values).astype("float32")
        # self.x_test_num = self.test_data[num_cols].values.astype("float32")
        self.x_test_id = np.nan_to_num(self.test_data[id_cols].values)
        self.y_test = self.test_data["target"].values.astype("float32")

        scale = StandardScaler()
        self.x_train_num = scale.fit_transform(self.x_train_num)
        self.x_valid_num = scale.transform(self.x_valid_num)
        self.x_test_num = scale.transform(self.x_test_num)

    def test(self, model_path = None):
        if model_path is not None:
            self.load_model(model_path)
        test_dataset = MLP_Dataset(self.x_train_num[:], self.x_train_id[:], self.y_train[:])
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        test_predictions = []

        with torch.no_grad():
            rmspe_score = 0.
            for x_num, x_id, y in test_loader:
                y_pred = self.model(x_num, x_id)
                loss = rmspe(y_pred, y, "torch")
                test_predictions.append(y_pred.cpu().numpy())
                rmspe_score += loss.item()
            rmspe_score /= len(test_loader)
        print(f"MLP test RMSPE is {rmspe_score}")
        return np.concatenate(test_predictions)

    def plot_loss(self, train_losses, valid_losses):
        plt.ylim(0.0005, 0.0030)
        plt.xlim(0, self.epochs)
        plt.grid()
        plt.plot(train_losses, label = "Train Loss")
        plt.plot(valid_losses, label = "Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def train_and_eval(self, model_path = None):
        train_dataset = MLP_Dataset(self.x_train_num[:], self.x_train_id[:], self.y_train[:])
        valid_dataset = MLP_Dataset(self.x_valid_num[:], self.x_valid_id[:], self.y_valid[:])
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        valid_loader = DataLoader(valid_dataset, batch_size = self.batch_size, shuffle = False)

        torch.device(self.device)
        criterion = RMSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

        min_valid_loss = float("inf")

        train_losses, valid_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.
            for x_num, x_id, y in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(x_num, x_id)
                loss = criterion(y_pred, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            print(f"Epoch {epoch} train loss is {train_loss}")

            self.model.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_rmpse_score = 0.
                for x_num, x_id, y in valid_loader:
                    y_pred = self.model(x_num, x_id)
                    loss = criterion(y_pred, y)
                    rmpse_score = rmspe(y_pred, y, "torch")
                    valid_loss += loss.item()
                    valid_rmpse_score += rmpse_score.item()
                valid_loss /= len(valid_loader)
                valid_rmpse_score /= len(valid_loader)
                valid_losses.append(valid_loss)
                print(f"Epoch {epoch} valid loss is {valid_loss}")
                print(f"Epoch {epoch} valid rmpse score is {valid_rmpse_score}")
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    if model_path is not None:
                        self.save_model(model_path)
        self.plot_loss(train_losses, valid_losses)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))