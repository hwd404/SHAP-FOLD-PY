import shap
import xgboost
from shap_fold import *
from timeit import default_timer as timer
from datetime import timedelta


def main():
    # data, attrs = load_data('data/acute/file1.csv')
    # data, attrs = load_data('data/autism/file1.csv')
    # data, attrs = load_data('data/breastw/file1.csv')
    # data, attrs = load_data('data/cars/file1.csv')
    # data, attrs = load_data('data/credit/file1.csv')
    # data, attrs = load_data('data/ecoli/file1.csv')
    # data, attrs = load_data('data/heart/file1.csv')
    # data, attrs = load_data('data/ionosphere/file1.csv')
    # data, attrs = load_data('data/kidney/file1.csv')
    # data, attrs = load_data('data/krkp/file1.csv')
    # data, attrs = load_data('data/mushroom/file1.csv')
    # data, attrs = load_data('data/sonar/file1.csv')
    # data, attrs = load_data('data/voting/file1.csv')
    # data, attrs = load_data('data/wine/file1.csv')

    # data, attrs = load_data('data/acute/file2.csv')
    # data, attrs = load_data('data/autism/file2.csv')
    # data, attrs = load_data('data/breastw/file2.csv')
    # data, attrs = load_data('data/cars/file2.csv')
    data, attrs = load_data('data/credit/file2.csv')
    # data, attrs = load_data('data/ecoli/file2.csv')
    # data, attrs = load_data('data/heart/file2.csv')
    # data, attrs = load_data('data/ionosphere/file2.csv')
    # data, attrs = load_data('data/kidney/file2.csv')
    # data, attrs = load_data('data/krkp/file2.csv')
    # data, attrs = load_data('data/mushroom/file2.csv')
    # data, attrs = load_data('data/sonar/file2.csv')
    # data, attrs = load_data('data/voting/file2.csv')
    # data, attrs = load_data('data/wine/file2.csv')
    # data, attrs = load_data('data/adult/file2.csv')
    # data, attrs = load_data('data/credit_card/file2.csv')


    X, Y = split_xy(data)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.8)

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


if __name__ == '__main__':
    main()
