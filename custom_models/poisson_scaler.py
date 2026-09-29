import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class PoissonScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, offset: float = 1e-3):
        self.offset = offset

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        X_ = X / np.sqrt(self.mean_ + self.offset)
        self.center_ = np.mean(X_, axis=0)
        return self

    def transform(self, X):
        X_ = X / np.sqrt(self.mean_ + self.offset)
        return X_ - self.center_
