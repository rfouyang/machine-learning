import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb


def get_data_iris():
    info = sklearn.datasets.load_iris()
    xs = info['data']
    ys = info['target']

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=0.2)

    return xtrain, xtest, ytrain, ytest

def get_data_breast_cancer():
    info = sklearn.datasets.load_breast_cancer()
    xs = info['data']
    ys = info['target']

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=0.2)

    return xtrain, xtest, ytrain, ytest

def get_data_boston():
    info = sklearn.datasets.load_boston()
    xs = info['data']
    ys = info['target']

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=0.2)

    return xtrain, xtest, ytrain, ytest

def get_data_boston_with_fname():
    info = sklearn.datasets.load_boston()
    xs = info['data']
    ys = info['target']
    fname = info['feature_names']

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(xs, ys, test_size=0.2)

    return xtrain, xtest, ytrain, ytest, fname


def multiclassification_xgboost():
    params = dict()
    params['eta'] = 0.3
    params['min_child_weight'] = 10
    params['cosample_bytree'] = 0.8
    params['max_depth'] = 3
    params['subsample'] = 0.5
    params['gamma'] = 2.0
    params['alpha'] = 1.0

    config = {
        'eval_metric': 'mlogloss',
        'objective': 'multi:softmax',
        'num_class': 3,
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1
    }

    config = {**config, **params}

    xtrain, xtest, ytrain, ytest = get_data_iris()


    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dtest = xgb.DMatrix(xtest, label=ytest)
    dpred = xgb.DMatrix(xtest)

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    num_boost_round = 1000
    model = xgb.train(config, dtrain, num_boost_round, evals=evallist, early_stopping_rounds=30, verbose_eval=True)

    ypred = model.predict(dpred)

    for yt, yp in zip(ytest, ypred):
        print(yt, yp)

    accuracy = sklearn.metrics.accuracy_score(ytest, ypred)
    print(accuracy)


def binaryclassification_xgboost():
    params = dict()
    params['eta'] = 0.3
    params['min_child_weight'] = 10
    params['cosample_bytree'] = 0.8
    params['max_depth'] = 3
    params['subsample'] = 0.5
    params['gamma'] = 2.0
    params['alpha'] = 1.0

    config = {
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1
    }

    config = {**config, **params}

    xtrain, xtest, ytrain, ytest = get_data_breast_cancer()


    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dtest = xgb.DMatrix(xtest, label=ytest)
    dpred = xgb.DMatrix(xtest)

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    num_boost_round = 1000
    model = xgb.train(config, dtrain, num_boost_round, evals=evallist, early_stopping_rounds=30, verbose_eval=True)

    ypred = model.predict(dpred)

    for yt, yp in zip(ytest, ypred):
        print(yt, yp)

    auc = sklearn.metrics.roc_auc_score(ytest, ypred)
    print(auc)

def regression_xgboost():
    params = dict()
    params['eta'] = 0.3
    params['min_child_weight'] = 10
    params['cosample_bytree'] = 0.8
    params['max_depth'] = 3
    params['subsample'] = 0.5
    params['gamma'] = 2.0
    params['alpha'] = 1.0

    config = {
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'silent': 1
    }

    config = {**config, **params}

    xtrain, xtest, ytrain, ytest = get_data_boston()


    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dtest = xgb.DMatrix(xtest, label=ytest)
    dpred = xgb.DMatrix(xtest)

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    num_boost_round = 1000
    model = xgb.train(config, dtrain, num_boost_round, evals=evallist, early_stopping_rounds=30, verbose_eval=True)

    ypred = model.predict(dpred)

    for yt, yp in zip(ytest, ypred):
        print(yt, yp)

    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    print(mse)

def feature_importance_xgboost():
    params = dict()
    params['eta'] = 0.3
    params['min_child_weight'] = 10
    params['cosample_bytree'] = 0.8
    params['max_depth'] = 5
    params['subsample'] = 0.5
    params['gamma'] = 2.0
    params['alpha'] = 1.0

    config = {
        'eval_metric': 'rmse',
        'objective': 'reg:linear',
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1
    }

    config = {**config, **params}

    xtrain, xtest, ytrain, ytest, fname = get_data_boston_with_fname()

    dtrain = xgb.DMatrix(xtrain, label=ytrain)
    dtest = xgb.DMatrix(xtest, label=ytest)

    evallist = [(dtrain, 'train'), (dtest, 'test')]
    num_boost_round = 10
    model = xgb.train(config, dtrain, num_boost_round, evals=evallist, early_stopping_rounds=100, verbose_eval=True)

    fmap_fp = 'fmap.txt'
    f = open(fmap_fp, 'w')
    for i, feature in enumerate(fname):
        f.write('{0}\t{1}\tq\n'.format(i, feature))
    f.close()

    feature_weights = model.get_fscore(fmap=fmap_fp)
    feature_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
    print(feature_weights)

    model.save_model('model.bin')
    model.dump_model('desc.txt', fmap=fmap_fp)

    xgb.plot_tree(model, fmap=fmap_fp, num_trees=2)
    plt.show()


def main():
    #regression_xgboost()
    #binaryclassification_xgboost()
    #multiclassification_xgboost()
    feature_importance_xgboost()

if __name__ == '__main__':
    main()