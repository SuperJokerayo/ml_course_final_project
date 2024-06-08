import torch
import numpy as np

def rmspe(y_true, y_pred, flag = "numpy"):
    if flag == "torch":
        return torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_rmspe_xgb(y_pred, trainer):
    y_true = trainer.get_label()
    return "RMSPE", rmspe(y_true, y_pred)

def feval_rmspe_lgb(y_pred, trainer):
    y_true = trainer.get_label()
    return "RMSPE", rmspe(y_true, y_pred), False