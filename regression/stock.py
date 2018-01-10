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
    path = os.path.join(BASE_DIR, 'stock.csv')
    df = pd.read_csv(path, engine='python')
    xs = df['CSI300'].values
    ys = df['GSYH'].values

    xs /= xs[0]
    ys /= ys[0]
    xs = np.log(xs)
    ys = np.log(ys)

    if show == True:
        plt.figure()
        plt.plot(xs, '-b')
        plt.plot(ys, '-r')
        plt.show()

        plt.figure()
        plt.scatter(xs, ys)
        plt.show()

    return xs, ys


def demo_capm():
    xs, ys = get_data()
    xs = np.atleast_2d(xs).T
    model = sklearn.linear_model.LinearRegression()
    model.fit(xs, ys)
    print('w:', model.coef_, 'b:', model.intercept_)

    ypred = model.predict(xs)

    plt.figure()
    plt.scatter(xs, ys)
    plt.plot(xs, ypred, '-r')
    plt.show()

def demo_trend():
    xs, ys = get_data()
    ts = np.atleast_2d(range(len(xs))).T

    model = sklearn.linear_model.LinearRegression()
    model.fit(ts, xs)
    print('w:', model.coef_, 'b:', model.intercept_)

    ypred = model.predict(ts)
    sig = np.sqrt(xs.var())
    plt.figure()
    plt.plot(ts, xs, '-b')
    plt.plot(ts, ypred + 0.5 * sig, '-k')
    plt.plot(ts, ypred, '-r')
    plt.plot(ts, ypred - 0.5 * sig, '-k')
    plt.show()

def main():
    demo_capm()
    demo_trend()


if __name__=='__main__':
    main()