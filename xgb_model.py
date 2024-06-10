import xgboost as xgb

import numpy as np
import matplotlib.pyplot as plt

from losses import rmspe, feval_rmspe_xgb

from base_model import Base_Model

class XGB_Model(Base_Model):
    def __init__(
        self,
        train_data,
        valid_data,
        test_data,
        hyperparams_path
    ):
        super(XGB_Model, self).__init__(
            train_data,
            valid_data,
            test_data,
            hyperparams_path
        )
        self._generate_feature_and_target()

    def _generate_feature_and_target(self):
        self.features = [col for col in self.train_data.columns if col not in {"target", "row_id"}]
        self.x_train = self.train_data[self.features]
        self.y_train = self.train_data["target"]
        self.x_valid = self.valid_data[self.features]
        self.y_valid = self.valid_data["target"]
        self.x_test = self.test_data[self.features]
        self.y_test = self.test_data["target"]

    def test(self, model_path = None):
        if model_path is not None:
            self.load_model(model_path)
        test_dataset = xgb.DMatrix(self.x_test)
        test_predictions = self.model.predict(test_dataset)
        rmspe_score = rmspe(self.y_test, test_predictions)
        print(f"XGB RMSPE is {rmspe_score}")
        # lgbm.plot_importance(self.model)
        return test_predictions

    def plot_loss(self, evals_result):
        train_loss = evals_result["train"]["rmse"]
        valid_loss = evals_result["valid"]["rmse"]
        plt.plot(train_loss, label = "Train Loss")
        plt.plot(valid_loss, label = "Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("rmse")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def train_and_eval(self):
        train_dataset = xgb.DMatrix(self.x_train, label = self.y_train)
        valid_dataset = xgb.DMatrix(self.x_valid, label = self.y_valid)

        evals_result = {}

        self.model = xgb.train(
            params = self.hyperparams,
            dtrain = train_dataset,
            num_boost_round = 5000,
            evals = [(train_dataset, "train"), (valid_dataset, "valid")],
            verbose_eval = 250,
            custom_metric = feval_rmspe_xgb,
            evals_result = evals_result
        )
        self.save_model("xgb_model.pth")
        self.plot_loss(evals_result)

    def save_model(self, model_path):
        self.model.save_model(model_path)

    def load_model(self, model_path):
        self.model = xgb.Booster(model_file = model_path)