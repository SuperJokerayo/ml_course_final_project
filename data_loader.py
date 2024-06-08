import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import get_time_stock, pca, processor
from xgb_model import XGB_Model
from lgb_model import LGB_Model
from mlp_model import MLP_Model


def load_data(data_dir):
    train = pd.read_csv(data_dir + "train.csv")
    train["row_id"] = train["stock_id"].astype(str) + '-' + train["time_id"].astype(str)
    train_ids = train.stock_id.unique()

    train_features = processor(data_dir, train_ids, is_train = True)
    train_features = train.merge(train_features, on = ["row_id"], how = "left")
    train_features = get_time_stock(train_features)
    pca(train_features)
    # train_features.to_csv("train_features.csv", index = False)

    # test = pd.read_csv(data_dir + "test.csv")
    # test["row_id"] = test["stock_id"].astype(str) + '-' + test["time_id"].astype(str)
    # test_ids = test.stock_id.unique()
#
    # test_features = processor(data_dir, test_ids, is_train = False)
    # test_features = test.merge(test_features, on = ["row_id"], how = "left")
    # test_features = get_time_stock(test_features)
    # test_features.to_csv("test_features.csv", index = False)
    train_features, test_features = train_test_split(train_features, train_size = 0.9, random_state = None)
    train_features, valid_features = train_test_split(train_features, train_size = 8. / 9., random_state = None)
    return train_features, valid_features, test_features

def load_features(data_dir):
    train_features = pd.read_csv(data_dir + "train_features.csv")
    test_features = pd.read_csv(data_dir + "test_features.csv")
    return train_features, test_features


if __name__ == "__main__":
    data_dir = "./data/optiver-realized-volatility-prediction/"
    # data_dir = "./"
    train_data, valid_data, test_data = load_data(data_dir)

    # hyperparams_path = "lgb_config.yaml"
    # model = LGB_Model(train_data, valid_data, test_data, hyperparams_path)
    # model.train_and_eval()
    # model.test()

    # hyperparams_path = "mlp_config.yaml"
    # model = MLP_Model(train_data, valid_data, test_data, hyperparams_path)
    # model.train_and_eval()

    hyperparams_path = "xgb_config.yaml"
    model = XGB_Model(train_data, valid_data, test_data, hyperparams_path)
    model.train_and_eval()
    model.test()