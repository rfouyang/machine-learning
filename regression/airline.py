import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.utils
import sklearn.model_selection
import sklearn.metrics
import sklearn.datasets
import sklearn.preprocessing


def get_data(show=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, 'airline.csv')
    df = pd.read_csv(path, engine='python')
    xs = df['Total'].values

    if show==True:
        plt.figure()
        plt.plot(xs,'-r')
        plt.show()

    return xs


def transform_data(xs, lag):
    n = len(xs)

    zs = []
    ys = []
    for i in range(lag, n-1):
        z = xs[i-lag: i]
        y = xs[i+1]
        zs.append(z)
        ys.append(y)

    zs = np.array(zs)
    ys = np.array(ys)
    return zs, ys


def demo_airline():
    series = get_data()
    lag = 12
    xs, ys = transform_data(series, lag)

    num, dim = xs.shape
    half = int(num / 2)
    xtrain, ytrain = xs[:half], ys[:half]
    xtest, ytest = xs[half:], ys[half:]

    model = sklearn.linear_model.LinearRegression()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    r2 = sklearn.metrics.r2_score(ytest, ypred)
    print(mse, r2)

    res = np.concatenate([ytrain, ypred])

    plt.figure()
    plt.plot(res, '-r')
    plt.plot(series[lag + 1:], '-b')
    plt.show()


def main():
    demo_airline()


if __name__=='__main__':
    main()
