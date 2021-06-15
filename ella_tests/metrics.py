import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

class Metrics:
    
    @staticmethod
    def r2(true, pred):
        return r2_score(true, pred)
    @staticmethod
    def rmse(true, pred):
        return np.sqrt(mean_squared_error(true, pred))
    @staticmethod
    def mae(true, pred):
        return mean_absolute_error(true, pred)
    @staticmethod
    def pearson(true, pred):
        true, pred = np.squeeze(true), np.squeeze(pred)
        return pearsonr(true, pred)
    @staticmethod
    def spearman(true, pred):
        true, pred = np.squeeze(true), np.squeeze(pred)
        return spearmanr(true, pred)

    def compute_metrics(self, true, pred, kinds):
        metrics = {}
        for kind in kinds:
            try:
                fn = getattr(self, kind)
            except NameError as e:
                print(e)
            error = fn(true, pred)
            metrics[kind] = error
        return metrics
