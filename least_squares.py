from numpy import ones, append, matmul
from numpy.linalg import inv
from sklearn.datasets import load_boston as load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def fit(x_train, y_train):
    intercept = ones(shape=(len(x_train), 1))
    x_train = append(intercept, x_train, axis=1)
    part1 = inv(matmul(x_train.T, x_train))
    part2 = matmul(x_train.T, y_train)
    return matmul(part1, part2)


def predict(x_test, coefficients):
    intercept = ones(shape=(len(x_test), 1))
    x_test = append(intercept, x_test, axis=1)
    return matmul(x_test, coefficients)


def mse(y_test, predictions):
    sum_ = 0
    for i in range(len(y_test)):
        sum_ += (y_test[i] - predictions[i]) ** 2
    return sum_ / len(y_test)


data = load().data
x_train, x_test, y_train, y_test = train_test_split(data, load().target, test_size=0.3)
coefficients = fit(x_train, y_train)
predictions = predict(x_test, coefficients)
print(mean_absolute_error(y_test, predictions))

reg = LinearRegression(fit_intercept=True, normalize=False)
reg.fit(x_train, y_train)
predictions = reg.predict(x_test)
print(mean_absolute_error(y_test, predictions))

poly = PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_test = poly.fit_transform(x_test)
reg = LinearRegression(fit_intercept=True, normalize=False)

reg.fit(x_train, y_train)
predictions = reg.predict(x_test)
print(mean_absolute_error(y_test, predictions))


# from numpy import ones, append, matmul
# from numpy.linalg import inv
# from sklearn.datasets import load_boston as load
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import scale, PolynomialFeatures
# from sklearn.linear_model import LinearRegression
#
# class Rergerssion:
#     def __init__(self):
#         self.x_train = x_train
#         self.y_train = y_train
#     def fit(self,x_train, y_train):
#         intercept = ones(shape=(len(self.x_train), 1))
#         x_train = append(intercept, x_train, axis=1)
#         part1 = inv(matmul(x_train.T, x_train))
#         part2 = matmul(x_train.T, y_train)
#         return matmul(part1, part2)
#
#
#     def predict(self,x_test, coefficients):
#         intercept = ones(shape=(len(x_test), 1))
#         x_test = append(intercept, x_test, axis=1)
#         return matmul(x_test, coefficients)
#
#
#     def mse(self,y_test, predictions):
#         sum_ = 0
#         for i in range(len(y_test)):
#             sum_ += (y_test[i] - predictions[i]) ** 2
#         return sum_ / len(y_test)
#
#
# data = load().data
# x_train, x_test, y_train, y_test = train_test_split(data, load().target, test_size=0.3)
# coefficients = Rergerssion()
# coefficients.fit(x_train, y_train)
# predictions = coefficients.predict(x_test, coefficients)
# print(mean_absolute_error(y_test, predictions))