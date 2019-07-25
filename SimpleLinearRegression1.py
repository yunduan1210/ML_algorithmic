import numpy as np

class SimpleLinearRegression1:

    def __init__(self):

        self.a = None
        self.b = None

    def fit(self, x_train, y_train):

        assert x_train.ndim == 1, \
            "simple Linear Regressor can only solve single feature training data."
        assert len(x_train)  == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a = num / d
        self.b = y_mean / self.a * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1,\
            "Simple Linear Regression can only solve single feature training data."
        assert self.a is not None and self.b is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测的数据x_single"""
        return self.a * x_single + self.b

    def __repr__(self):
        return "SimpleLinearRegression1()"