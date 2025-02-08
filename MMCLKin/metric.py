import os
import random
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from scipy import stats
from lifelines.utils import concordance_index

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    pearson_value = np.corrcoef(y_true, y_pred)[0, 1]
    spear_value = spearman(y_true, y_pred)
    d = {'mae': mae, 'ci': ci, 'mse': mse, 'rmse': rmse, 'r2': r2, 'pearson_value':pearson_value, "spearman":spear_value}
    return d



