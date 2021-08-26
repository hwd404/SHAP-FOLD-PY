import numpy as np
import xgboost
import shap
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from datetime import timedelta
import one_hot_encoding as oh
import decision_tree_encoding as dt
import shap_fold as sf


def preprocess():
    # columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jaundice','autism','used_app_before','relation']
    # attrs, data = convert_data('data/autism/autism.csv', columns, 'label', 'YES', numerics=['age'])
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    attrs, data = oh.convert_data('data/cars/cars.csv', columns, 'label', 'positive', numerics=['a1'])
    res = [attrs]
    for d in data:
        res.append(d)
    # f = open('data/autism/file.csv', 'w')
    f = open('data/cars/file.csv', 'w')
    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()


def preprocess2():
    # columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jundice','autism','used_app_before','relation']
    # data, num_idx = load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=['age'], pos='YES')
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    data, num_idx = dt.load_data('data/cars/cars.csv', attrs=columns, label=['label'], numerics=['a1'], pos='positive')
    _, n = np.shape(data)
    res = dt.encode_data(data, num_idx, columns)
    # f = open('data/autism/file2.csv', 'w')
    f = open('data/cars/file2.csv', 'w')
    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()


def main():
    # preprocess()
    preprocess2()

    # data, attrs = sf.load_data('data/autism/file.csv')
    # data, attrs = sf.load_data('data/autism/file2.csv')
    # data, attrs = sf.load_data('data/cars/file.csv')
    data, attrs = sf.load_data('data/cars/file2.csv')

    X, Y = sf.split_xy(data)
    data_train, data_test = sf.split_set(data, 0.8)
    data_train, data_valid = sf.split_set(data_train, 0.5)
    X_train, Y_train = sf.split_xy(data_train)
    X_valid, Y_valid = sf.split_xy(data_valid)
    valid_set = [(X_valid, Y_valid)]
    model = xgboost.XGBClassifier(objective='binary:logistic',
                                  use_label_encoder=False,
                                  n_estimators=10,
                                  max_depth=3,
                                  gamma=1,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  learning_rate=0.1
                                  ).fit(X_train, Y_train, early_stopping_rounds=100, eval_set=valid_set,
                                        eval_metric=["logloss"])

    X_test, Y_test = sf.split_xy(data_test)
    Y_test_hat = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_test_hat)
    print('xgb model accuracy: ', accuracy)

    Y_train_hat = model.predict(X_train)
    explainer = shap.Explainer(model)
    X_pos, X_neg = sf.split_X_by_Y(X_train, Y_train_hat)
    # print('X_pos', X_pos, 'Y_train_hat', Y_train_hat)
    SHAP_pos = sf.get_shap(explainer, X_pos)
    SHAP_neg = sf.get_shap(explainer, X_neg)

    start = timer()
    depth = -1
    rules = sf.shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, depth=depth)

    # print(rules, len(rules))
    fidelity, _, _, _ = sf.get_metrics(rules, X_test, Y_test_hat)
    accuracy, _, _, _ = sf.get_metrics(rules, X_test, Y_test)
    print('fidelity: ', fidelity, 'accuracy: ', accuracy)

    frules = sf.flatten_rules(rules)
    drules = sf.decode_rules(frules, attrs)
    print(drules, len(drules))

    end = timer()
    print('total costs: ', timedelta(seconds=end - start))


if __name__ == '__main__':
    main()
