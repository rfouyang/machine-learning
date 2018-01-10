import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.utils
import sklearn.model_selection
import sklearn.metrics
import sklearn.datasets
import sklearn.preprocessing
import tensorflow as tf


def get_data(show=False):
    info = sklearn.datasets.load_boston()
    xs = info['data']
    ys = info['target']

    if show==True:
        df = pd.DataFrame(data=xs, columns=info['feature_names'])
        print(df.head())
        print(df.describe())

    scaler = sklearn.preprocessing.StandardScaler()
    xs = scaler.fit_transform(xs)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=0.1, random_state=2018)
    return xtrain, xtest, ytrain, ytest


def regression_sklearn():
    xtrain, xtest, ytrain, ytest = get_data(show=False)

    model = sklearn.linear_model.LinearRegression()
    model.fit(xtrain, ytrain)

    print('w:', model.coef_, 'b:', model.intercept_)
    ypred = model.predict(xtest)

    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    r2 = sklearn.metrics.r2_score(ytest, ypred)
    print(mse, r2)

    plt.figure()
    plt.scatter(ytest, ypred)
    plt.show()


def regression_tensorflow():
    tf.set_random_seed(2018)
    xtrain, xtest, ytrain, ytest = get_data()
    ytrain = np.atleast_2d(ytrain).T
    ytest = np.atleast_2d(ytest).T

    num, dim = xtrain.shape

    print(xtrain.shape)
    print(ytrain.shape)

    input = tf.placeholder(tf.float32, shape=[None, dim], name='input')
    target = tf.placeholder(tf.float32, shape=[None, 1], name='target')

    W = tf.Variable(tf.random_normal([dim, 1], stddev=0.01), name='W')
    b = tf.Variable(tf.random_normal([1], stddev=0.01), name='b')

    pred = tf.matmul(input, W) + b

    loss = tf.reduce_mean(tf.square(target - pred))
    train_op = tf.train.AdamOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        epoches = 10000
        for epoch in range(epoches):
            pred_eval, loss_eval, _ = sess.run([pred, loss, train_op], feed_dict={input: xtrain, target: ytrain})

            print('\repoch: {} loss:{}'.format(epoch, loss_eval), end="")

        print('\nW:', W.eval(), 'b:', b.eval())

        pred_eval = sess.run(pred, feed_dict={input: xtest, target: ytest})

        mse = sklearn.metrics.mean_squared_error(ytest, pred_eval)
        r2 = sklearn.metrics.r2_score(ytest, pred_eval)
        print(mse, r2)

        plt.figure()
        plt.scatter(ytest, pred_eval)
        plt.show()

def main():
    #regression_sklearn()
    regression_tensorflow()

if __name__=='__main__':
    main()