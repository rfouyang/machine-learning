import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.utils
import tensorflow as tf


def gen_data1d(show=False):
    np.random.seed(2018)

    n = 100
    xs = np.linspace(-1.0, 1.0, n)
    ys = 2 * xs + 0.2 + 0.4 * np.random.randn(n)
    yt = 2 * xs + 0.2

    if show==True:
        plt.figure()
        plt.scatter(xs, ys)
        plt.plot(xs, yt, '-r')
        plt.show()

    return xs, ys


def regression_sklearn():
    xs, ys = gen_data1d(show=False)
    xs = np.atleast_2d(xs).T

    model = sklearn.linear_model.LinearRegression()
    model.fit(xs, ys)

    print('w:', model.coef_, 'b:', model.intercept_)

    n = 100
    xt = np.atleast_2d(np.linspace(-1.0, 1.0, n)).T
    yp = model.predict(xt)

    plt.figure()
    plt.scatter(xs, ys)
    plt.plot(xt,yp, '-r')
    plt.show()


def regression_tensorflow():
    xs, ys = gen_data1d()
    xs = np.atleast_2d(xs).T
    ys = np.atleast_2d(ys).T

    n = 100
    xt = np.atleast_2d(np.linspace(-1.0, 1.0, n)).T

    input = tf.placeholder(tf.float32, shape=[None, 1], name='input')
    target = tf.placeholder(tf.float32, shape=[None, 1], name='target')

    w = tf.Variable(tf.random_normal([1, 1], stddev=0.35), name='w')
    b = tf.Variable(tf.random_normal([1], stddev=0.35), name='b')

    pred = tf.matmul(input, w) + b
    loss = tf.reduce_mean(tf.pow(target - pred, 2))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        epoches = 10000
        for epoch in range(epoches):
            sess.run(train_op, feed_dict={input: xs, target: ys})
        print('w:', w.eval(), 'b:', b.eval())

        pred_eval = sess.run(pred, feed_dict={input: xt, target: ys})

        plt.figure()
        plt.scatter(xs, ys)
        plt.plot(xt, pred_eval, '-r')
        plt.show()


def main():
    #gen_data1d(show=True)
    #regression_sklearn()
    regression_tensorflow()


if __name__=='__main__':
    main()