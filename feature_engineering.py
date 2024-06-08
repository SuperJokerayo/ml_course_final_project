import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 计算加权平均价格
def calc_wap(df, rank = 1):
    wap = (df[f"bid_price{rank}"] * df[f"ask_size{rank}"] + df[f"ask_price{rank}"] * df[f"bid_size{rank}"]) / (df[f"bid_size{rank}"] + df[f"ask_size{rank}"])
    return wap

# 计算对数回报率
def log_return(series):
    return np.log(series).diff()

# 计算实际波动率
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def count_unique(series):
    return len(np.unique(series))

def get_stats_window(df, operations, seconds_in_bucket = 0):
    local_df_feature = df[df["seconds_in_bucket"] >= seconds_in_bucket].groupby(["time_id"]).agg(operations).reset_index()
    local_df_feature.columns = ["_".join(col) for col in local_df_feature.columns]
    if seconds_in_bucket != 0:
        local_df_feature = local_df_feature.add_suffix(f"_{seconds_in_bucket}")
    return local_df_feature

def tendency(price, vol):
    df_diff = np.diff(price)
    val = (df_diff / price[1:]) * 100
    power = np.sum(val * vol[1:])
    return (power)

def book_processor(df, stock_id = 0):
    df["wap1"] = calc_wap(df, 1)
    df["wap2"] = calc_wap(df, 2)

    df["log_return1"] = df.groupby(["time_id"])["wap1"].apply(log_return).values
    df["log_return2"] = df.groupby(["time_id"])["wap2"].apply(log_return).values

    # 计算wap差
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])

    # 计算价格差
    df["price_spread1"] = (df["ask_price1"] - df["bid_price1"]) / ((df["ask_price1"] + df["bid_price1"]) / 2)
    df["price_spread2"] = (df["ask_price2"] - df["bid_price2"]) / ((df["ask_price2"] + df["bid_price2"]) / 2)
    df["bid_spread"] = df["bid_price1"] - df["bid_price2"]
    df["ask_spread"] = df["ask_price1"] - df["ask_price2"]
    df["bid_ask_spread"] = abs(df["bid_spread"] - df["ask_spread"])

    # 计算总共的股数
    df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (df["bid_size1"] + df["bid_size2"])

    # 计算买卖差
    df["volume_imbalance"] = abs((df["ask_size1"] + df["ask_size2"]) - (df["bid_size1"] + df["bid_size2"]))

    base_features = [
        "wap1", "wap2", "log_return1", "log_return2",
        "wap_balance", "price_spread1", "price_spread2",
        "bid_spread", "ask_spread", "bid_ask_spread",
        "total_volume", "volume_imbalance"
    ]

    create_base_features_operations = {}

    for feature in base_features:
        create_base_features_operations[feature] = ["min", "max", "mean", "std", "sum"]

    create_base_features_operations["log_return1"].append(realized_volatility)
    create_base_features_operations["log_return2"].append(realized_volatility)

    create_time_features_operations = {
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility]
    }

    df_feature = None

    for i in range(0, 501, 100):
        if df_feature is None:
            df_feature = get_stats_window(
                df,
                create_base_features_operations,
                i
            )
        else:
            df_feature = df_feature.merge(
                get_stats_window(df, create_time_features_operations, i),
                how = "left",
                left_on = "time_id_",
                right_on = f"time_id__{i}"
            )
    df_feature.drop([f"time_id__{i}" for i in range(100, 501, 100)], axis = 1, inplace = True)
    df_feature = df_feature.add_prefix("book_")
    df_feature["row_id"] = df_feature["book_time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature.drop(["book_time_id_"], axis = 1, inplace = True)
    df_feature.insert(0, "row_id", df_feature.pop("row_id"))
    df_feature.to_csv("df_book_feature.csv", index = False)
    return df_feature

def trade_processor(df, stock_id = 0):
    df["log_return"] = df.groupby(["time_id"])["price"].apply(log_return).values
    df["amount"] = df["price"] * df["size"]

    create_base_features_operations = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": [count_unique],
        "size": ["min", "max", "mean", "std", "sum"],
        "order_count": ["min", "max", "mean", "std", "sum"],
        "amount": ["min", "max", "mean", "std", "sum"]
    }

    create_time_features_operations = create_base_features_operations.copy()
    create_time_features_operations.pop("amount")

    lis = []
    for n_time_id in df["time_id"].unique():
        df_id = df[df['time_id'] == n_time_id]
        tendencyV = tendency(df_id["price"].values, df_id["size"].values)
        f_max = np.sum(df_id["price"].values > np.mean(df_id["price"].values))
        f_min = np.sum(df_id["price"].values < np.mean(df_id["price"].values))
        df_max = np.sum(np.diff(df_id["price"].values) > 0)
        df_min = np.sum(np.diff(df_id["price"].values) < 0)

        abs_diff = np.median(np.abs(df_id["price"].values - np.mean(df_id["price"].values)))
        energy = np.mean(df_id["price"].values ** 2)
        iqr_p = np.percentile(df_id["price"].values, 75) - np.percentile(df_id["price"].values, 25)

        abs_diff_v = np.median(np.abs(df_id["size"].values - np.mean(df_id["size"].values)))
        energy_v = np.sum(df_id["size"].values ** 2)
        iqr_p_v = np.percentile(df_id["size"].values, 75) - np.percentile(df_id["size"].values, 25)

        lis.append({
            "time_id": n_time_id, "tendency": tendencyV,
            "f_max": f_max, "f_min": f_min, "df_max": df_max,
            "df_min": df_min, "abs_diff": abs_diff, "energy": energy,
            "iqr_p": iqr_p, "abs_diff_v": abs_diff_v,
            "energy_v": energy_v, "iqr_p_v": iqr_p_v
        })

    df_lr = pd.DataFrame(lis)
    df_feature = None
    for i in range(0, 501, 100):
        if df_feature is None:
            df_feature = get_stats_window(
                df,
                create_base_features_operations,
                i
            )
            df_feature = df_feature.merge(
                df_lr,
                how = "left",
                left_on = "time_id_",
                right_on = "time_id"
            )
        else:
            df_feature = df_feature.merge(
                get_stats_window(df, create_time_features_operations, i),
                how = "left",
                left_on = "time_id_",
                right_on = f"time_id__{i}"
            )
    df_feature.drop([f"time_id__{i}" for i in range(100, 501, 100)], axis = 1, inplace = True)
    df_feature = df_feature.add_prefix("trade_")
    df_feature["row_id"] = df_feature["trade_time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature.drop(["trade_time_id_"], axis = 1, inplace = True)
    df_feature.insert(0, "row_id", df_feature.pop("row_id"))
    df_feature.to_csv("df_trade_feature.csv", index = False)
    return df_feature

def get_time_stock(df):
    vol_cols = [
        "book_log_return1_realized_volatility",
        "book_log_return2_realized_volatility",
        "book_log_return1_realized_volatility_400",
        "book_log_return2_realized_volatility_400",
        "book_log_return1_realized_volatility_300",
        "book_log_return2_realized_volatility_300",
        "book_log_return1_realized_volatility_200",
        "book_log_return2_realized_volatility_200",
        "trade_log_return_realized_volatility",
        "trade_log_return_realized_volatility_400",
        "trade_log_return_realized_volatility_300",
        "trade_log_return_realized_volatility_200"
    ]

    df_stock_id = df.groupby(["stock_id"])[vol_cols].agg(["min", "max", "mean", "std", "sum"]).reset_index()
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix("_stock")


    df_time_id = df.groupby(["time_id"])[vol_cols].agg(["min", "max", "mean", "std", "sum"]).reset_index()
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix("_time")

    df = df.merge(df_stock_id, how = "left", left_on = ["stock_id"], right_on = ["stock_id__stock"])
    df = df.merge(df_time_id, how = "left", left_on = ["time_id"], right_on = ["time_id__time"])
    df.drop(["stock_id__stock", "time_id__time"], axis = 1, inplace = True)
    return df

def pca(df, n_pca = 0.9):
    exclusion = ["time_id", "target", "row_id", "stock_id"]
    features = [col for col in df.columns if col not in exclusion]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[features])
    X_num = np.nan_to_num(X_num, posinf = 0, neginf = 0)
    _pca = PCA(n_components = n_pca, random_state = 0)
    X_num = pd.DataFrame(_pca.fit_transform(X_num))
    X_num.fillna(X_num.mean())
    return pd.concat([df[exclusion], X_num], axis = 1)

def processor(data_dir, stock_ids, is_train = True):
    def p(stock_id):
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)
        book_df = pd.read_parquet(file_path_book)
        trade_df = pd.read_parquet(file_path_trade)
        df_single_stock = pd.merge(
            book_processor(book_df, stock_id),
            trade_processor(trade_df, stock_id),
            on = "row_id", how = "left"
        )
        return df_single_stock

    features = Parallel(n_jobs = -1, verbose = 1)(delayed(p)(stock_id) for stock_id in tqdm(stock_ids))
    features = pd.concat(features, ignore_index = True)
    return features


if __name__ == "__main__":
    data_dir = "./data/optiver-realized-volatility-prediction/"
    train = pd.read_csv(data_dir + "train.csv")
    train["row_id"] = train["stock_id"].astype(str) + '-' + train["time_id"].astype(str)
    train_ids = train.stock_id.unique()

    train_features = processor(data_dir, train_ids, is_train = True)
    train_features = train.merge(train_features, on = ["row_id"], how = "left")
    train_features = get_time_stock(train_features)
    train_features.to_csv("train_features.csv", index = False)

    test = pd.read_csv(data_dir + "test.csv")
    test["row_id"] = test["stock_id"].astype(str) + '-' + test["time_id"].astype(str)
    test_ids = test.stock_id.unique()

    test_features = processor(data_dir, test_ids, is_train = False)
    test_features = test.merge(test_features, on = ["row_id"], how = "left")
    test_features = get_time_stock(test_features)
    test_features.to_csv("test_features.csv", index = False)
