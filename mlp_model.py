import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from base_model import Base_Model
from losses import rmspe

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
        self.hyperparams["src_num_dim"] = self.x_train_num.shape[1]
        self.model = MLP(self.hyperparams)
        self._init_hyperparams()

    def _init_hyperparams(self):
        self.epochs = self.hyperparams["epochs"]
        self.lr = self.hyperparams["lr"]
        self.batch_size = self.hyperparams["batch_size"]

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

    def test(self):
        test_dataset = MLP_Dataset(self.x_train_num[:], self.x_train_id[:], self.y_train[:])
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        with torch.no_grad():
            rmspe_score = 0.
            for x_num, x_id, y in test_loader:
                y_pred = self.model(x_num, x_id)
                loss = rmspe(y_pred, y, "torch")
                rmspe_score += loss.item()
            rmspe_score /= len(test_loader)
        print(f"MLP test RMSPE is {rmspe_score}")

    def train_and_eval(self):
        train_dataset = MLP_Dataset(self.x_train_num[:], self.x_train_id[:], self.y_train[:])
        valid_dataset = MLP_Dataset(self.x_valid_num[:], self.x_valid_id[:], self.y_valid[:])
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        valid_loader = DataLoader(valid_dataset, batch_size = self.batch_size, shuffle = False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

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
            print(f"Epoch {epoch} train loss is {train_loss / len(train_loader)}")
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0.
                for x_num, x_id, y in valid_loader:
                    y_pred = self.model(x_num, x_id)
                    loss = rmspe(y_pred, y, "torch")
                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                print(f"Epoch {epoch} valid rmpse score is {valid_loss}")