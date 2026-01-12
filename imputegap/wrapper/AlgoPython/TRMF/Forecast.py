# ===============================================================================================================
# SOURCE: https://github.com/SemenovAlex/trmf/tree/master
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
# ===============================================================================================================

"""
Different simple models for timeseries.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

class AutoRegressionModel():
    """ Forecasting with AR model for each timeseries with lags """
    def __init__(self, lags):
        self.lags = lags

    def fit(self, train):
        self.train = train
        self.coefs = []
        self.intercept = []

        for index in range(len(train)):
            series = train[index]
            features = [np.roll(series, shift=lag) for lag in self.lags]
            X = np.vstack([features])

            model = LinearRegression()
            model.fit(X[:,max(self.lags):].T, series[max(self.lags):])

            self.intercept.append(model.intercept_)
            self.coefs.append(model.coef_)

    def predict(self, h):
        preds = []
        for index in range(len(self.train)):
            series = self.train[index]
            for _ in range(h):
                features = np.array([series[-lag] for lag in self.lags])
                pred = (self.coefs[index] * features).sum() + self.intercept[index]
                series = np.append(series, pred)
            preds.append(series[-h:])
        return np.vstack(preds)


class NaiveModel():
    """ Forecasting by the last known value """

    def __init__(self):
        return

    def fit(self, train):
        self.train = train
        self.N, self.T = self.train.shape
        self.preds = np.array([])

        for index in range(self.N):
            t = self.T - 1
            while (t > 0) and (np.isnan(self.train[index][t])):
                t -= 1
            if t < 0:
                pred = 0
            else:
                pred = self.train[index][t]
            self.preds = np.append(self.preds, pred)

    def predict(self, h):
        return self.preds.repeat(h).reshape(self.N, h)

    def impute_missings(self):
        for index in range(self.N):
            last_value = 0
            for t in range(self.T):
                if np.isnan(self.train[index][t]):
                    self.train[index][t] = last_value
                last_value = self.train[index][t]
        return self.train


class MeanModel():
    """ Forecasting by the mean of all data """
    def __init__(self):
        return

    def fit(self, train):
        self.train = train
        self.N, self.T = self.train.shape
        self.preds = np.array([])

        for index in range(len(train)):
            series = self.train[index].copy()
            if (~np.isnan(series)).sum() == 0:
                pred = 0
            else:
                pred = series[~np.isnan(series)].mean()
            self.preds = np.append(self.preds, pred)

    def predict(self, h):
        return self.preds.repeat(h).reshape(self.N, h)

    def impute_missings(self):
        for index in range(self.N):
            for t in range(self.T):
                if np.isnan(self.train[index][t]):
                    self.train[index][t] = self.preds[index]
        return self.train

