import os
import time
from data_loader import load_data
from xgb_model import XGB_Model
from lgb_model import LGB_Model
from mlp_model import MLP_Model

models = {
    "xgb": XGB_Model,
    "lgb": LGB_Model,
    "mlp": MLP_Model
}
def run(train_data, valid_data, test_data, model_type):
    hyperparams_path = f"{model_type}_config.yaml"
    model = models[model_type](train_data, valid_data, test_data, hyperparams_path)
    model.train_and_eval()
    model_path = f"{model_type}_model.pth"
    if os.path.exists(model_path):
        model.test(model_path)
    else:
        model.test()

def main(data_dir, model_types = ["xgb", "lgb", "mlp"]):
    print("Running feature engineering ....")
    start = time.time()
    train_data, valid_data, test_data = load_data(data_dir)
    print(f"Feature engineering takes {time.time() - start} seconds")
    print()
    for model_type in model_types:
        print(f"Running {model_type} model ....")
        start = time.time()
        run(train_data, valid_data, test_data, model_type)
        print(f"{model_type} model takes {time.time() - start} seconds")
        print()
    print("Done")

if __name__ == "__main__":
    data_dir = "./data/optiver-realized-volatility-prediction/"
    main(data_dir, model_types = ["xgb"])