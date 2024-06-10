import warnings
warnings.filterwarnings("ignore")

import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from .losses import rmspe, feval_rmspe_lgb

from .base_model import Base_Model


class LGB_Model(Base_Model):
    def __init__(
        self,
        train_data,
        valid_data,
        test_data,
        hyperparams_path
    ):
        super(LGB_Model, self).__init__(
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
        test_predictions = self.model.predict(self.x_test)
        rmspe_score = rmspe(self.y_test, test_predictions)
        print(f"LGB RMSPE is {rmspe_score}")
        # lgb.plot_importance(self.model, max_num_features = 10)
        # plt.show()
        return test_predictions

    def plot_loss(self, valid_results):
        lgb.plot_metric(
            valid_results,
            metric = "rmse",
            xlim = (0, self.hyperparams["trainer"]["num_boost_round"]),
            ylim = (0.0005, 0.0030),
            xlabel = "Epochs",
            ylabel = "rmse",
            title = "Training and Validation Loss",
            grid = True
        )
        plt.legend(["Train Loss", "Validation Loss"])
        plt.show()

    def train_and_eval(self, model_path = None):
        train_dataset = lgb.Dataset(self.x_train, self.y_train)
        valid_dataset = lgb.Dataset(self.x_valid, self.y_valid)
        valid_results = {}
        self.model = lgb.train(
            params = self.hyperparams["model"],
            num_boost_round = self.hyperparams["trainer"]["num_boost_round"],
            train_set = train_dataset,
            valid_sets = [train_dataset, valid_dataset],
            verbose_eval = self.hyperparams["trainer"]["verbose_eval"],
            feval = feval_rmspe_lgb,
            evals_result = valid_results
        )
        if model_path is not None:
            self.save_model(model_path)
        self.plot_loss(valid_results)


    def save_model(self, model_path):
        self.model.save_model(model_path)

    def load_model(self, model_path):
        self.model = lgb.Booster(model_file = model_path)