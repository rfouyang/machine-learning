
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
        'tree_method': 'exact',
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

    mes = sklearn.metrics.mean_squared_error(ytest, ypred)
    print(mes)

def main():
    regression_xgboost()
    binaryclassification_xgboost()
    multiclassification_xgboost()


if __name__ == '__main__':
    main()