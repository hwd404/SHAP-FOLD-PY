import numpy as np
import shap
from shap_fold import *
import xgboost
from timeit import default_timer as timer
from datetime import timedelta
from scasp_utils import load_data_pred, load_translation, scasp_query


class Example(object):
    def __init__(self, X, Y):
        self.mem = dict()
        for i in range(len(Y)):
            k = tuple(X[i])
            if self.mem.get(k) is None:
                self.mem[k] = 0
            self.mem[k] += 1 if Y[i] else -1

    def classify(self, x):
        k = tuple(x)
        if self.mem.get(k) is None:
            return 0
        return 1 if self.mem[k] > 0 else 0

    def predict(self, X):
        ret = []
        for x in X:
            ret.append(self.classify(x))
        return np.array(ret)


def main():
    data, attrs = load_data('data/cars/file2.csv')

    X, Y = split_xy(data)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8)

    model = Example(X_train, Y_train)
    Y_train_hat = model.predict(X_train)
    X_pos, X_neg = split_X_by_Y(X_train, Y_train_hat)

    X_train = shap.sample(X_train, nsamples=100)
    explainer = shap.KernelExplainer(model.predict, X_train)
    SHAP_pos = explainer.shap_values(X_pos)
    SHAP_neg = explainer.shap_values(X_neg)

    model = Classifier(attrs=attrs)
    start = timer()
    model.fit(X_pos, SHAP_pos, X_neg, SHAP_neg)
    end = timer()

    model.print_asp()
    print('% # of rules: ', len(model.asp_rules))

    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% shap_fold costs: ', timedelta(seconds=end - start))


def titanic():
    data_train, attrs = load_data('data/titanic/file_train.csv')
    data_test, attrs = load_data('data/titanic/file_test.csv')
    X_train, Y_train = split_xy(data_train)
    X_test, Y_test = split_xy(data_test)

    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']
    X_pred = load_data_pred('data/titanic/test.csv', numerics=nums)

    model = xgboost.XGBClassifier(objective='binary:logistic',
                                  max_depth=3,
                                  n_estimators=10,
                                  use_label_encoder=False).fit(X_train, Y_train,
                                                               eval_metric=["logloss"])

    Y_train_hat = model.predict(X_train)
    X_pos, X_neg = split_X_by_Y(X_train, Y_train_hat)

    explainer = shap.Explainer(model)
    SHAP_pos = explainer(X_pos).values
    SHAP_neg = explainer(X_neg).values

    model = Classifier(attrs=attrs)
    start = timer()
    model.fit(X_pos, SHAP_pos, X_neg, SHAP_neg)
    end = timer()

    model.print_asp()
    print('% # of rules: ', len(model.asp_rules))

    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4), 'r', round(r, 4), 'f1', round(f1, 4))
    print('% shap_fold costs: ', timedelta(seconds=end - start))

    load_translation(model, 'data/titanic/template.txt')
    for i in range(len(X_test)):
        print(model.classify(X_test[i]))
        res = scasp_query(model, X_pred[i], pred=True)
        print(res)


if __name__ == '__main__':
    main()
    # titanic()
