from torchmetrics import R2Score
import numpy as np
import random
import torch


def Rmse(y, y_pred):

    return torch.sqrt(torch.mean(torch.square(y_pred - y)))


def r2(y, y_pred):

    y_mean = torch.sum(torch.mean(y, dim=0))
    rmses = Rmse(y, y_pred)
    score = 1 - (rmses**2)/(y_mean**2)

    return score