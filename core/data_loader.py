import pandas as pd
from sklearn.model_selection import train_test_split

from .feature_engineering import get_time_stock, pca, processor

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

