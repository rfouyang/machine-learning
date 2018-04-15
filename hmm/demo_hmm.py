__author__ = 'Ouyang Ruofei'
__email__ = 'ruofei@wecash.asia'


import numpy as np
from hmmlearn import hmm


def demo_hmmlearn_multi():
    hidden = ['sunny', 'cloudy', 'rainy']
    no_hidden = len(hidden)

    obs = ['dry', 'dryish', 'damp', 'soggy']
    no_obs = len(obs)

    pi = np.array([0.63, 0.17, 0.2])
    A = np.array([
        [0.5, 0.375, 0.125],
        [0.25, 0.125, 0.625],
        [0.25, 0.375, 0.375]
    ])

    B = np.array([
        [0.6, 0.2, 0.15, 0.05],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.1, 0.35, 0.5]
    ])

    model = hmm.MultinomialHMM(n_components=no_hidden, n_iter=10000, random_state=2018)
    model.startprob = pi
    model.transmat = A
    model.emissionprob = B

    # X shape: col by col matrix
    seq_obs = np.atleast_2d([
        [0, 0, 0, 1, 1, 1, 2, 3, 3],
        [3, 2, 1, 1, 1, 0, 0, 0, 0]
    ]).T

    print('\nuse estimated params:')
    model.fit(seq_obs)

    print(model.startprob_)
    print(model.transmat_)
    print(model.emissionprob_)

    seq_test = np.atleast_2d([0, 1, 2, 1, 0]).T
    logprob, seq_hidden = model.decode(seq_test, algorithm="viterbi")

    print(logprob)
    for z, x in zip(seq_hidden, seq_test[:, 0]):
        print('{}->{}'.format(hidden[z], obs[x]))


    print('\nuse empirical params:')

    model.startprob_ = model.startprob
    model.transmat_ = model.transmat
    model.emissionprob_ = model.emissionprob
    logprob, seq_hidden = model.decode(seq_test, algorithm="viterbi")

    print(model.startprob_)
    print(model.transmat_)
    print(model.emissionprob_)

    print(logprob)
    for z, x in zip(seq_hidden, seq_test[:, 0]):
        print('{}->{}'.format(hidden[z], obs[x]))


def main():
    demo_hmmlearn_multi()


if __name__ == '__main__':
    main()