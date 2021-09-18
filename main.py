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
    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    # attrs, data = oh.convert_data('data/acute/acute.csv', columns, 'label', 'yes', numerics=['a1'])

    columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jaundice','autism',
    'used_app_before','relation']
    attrs, data = oh.convert_data('data/autism/autism.csv', columns, 'label', 'YES', numerics=['age'])

    # columns = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    # 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    # attrs, data = oh.convert_data('data/breastw/breastw.csv', columns, 'label', 'malignant', numerics=columns)

    # columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    # attrs, data = oh.convert_data('data/cars/cars.csv', columns, 'label', 'positive', numerics=[])

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    # attrs, data = oh.convert_data('data/credit/credit.csv', columns, 'label', '+', numerics=['a2', 'a3', 'a8',
    # 'a11', 'a14', 'a15'])

    # columns = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    # 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina',
    # 'oldpeak', 'slope', 'major_vessels', 'thal']
    # attrs, data = oh.convert_data('data/heart/heart.csv', columns, 'label', 'present', numerics=['age',
    # 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak'])

    # columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
    # 'pcv','wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    # attrs, data = oh.convert_data('data/kidney/kidney.csv', columns, 'label', 'ckd', numerics=['age', 'bp', 'sg',
    # 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'])

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
    # 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30',
    # 'a31', 'a32', 'a33', 'a34', 'a35', 'a36']
    # attrs, data = oh.convert_data('data/krkp/krkp.csv', columns, 'label', 'won', numerics=[])

    # columns = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    # 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    # 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    # 'spore_print_color', 'population', 'habitat']
    # attrs, data = oh.convert_data('data/mushroom/mushroom.csv', columns, 'label', numerics=[], pos='p')

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
    # 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30',
    # 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45',
    # 'a46', 'a47', 'a48', 'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    # data, num_idx = oh.load_data('data/sonar/sonar.csv', attrs=columns, label=['label'], numerics=columns, pos='Mine')

    # columns = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    # 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    # 'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending',
    # 'superfund_right_to_sue', 'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    # attrs, data = oh.convert_data('voting.csv', columns, 'label', 'republican', numerics=[])

    # columns = ['Sex','Age','Number_of_Siblings_Spouses','Number_Of_Parents_Children','Fare','Class','Embarked']
    # attrs, data = oh.convert_data('data/titanic/tot.csv', columns, 'Survived', '1', numerics=['Age','Number_of_Siblings_Spouses','Number_Of_Parents_Children','Fare'])

    res = [attrs]
    for d in data:
        res.append(d)

    f = open('data/autism/file1.csv', 'w')

    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()


def preprocess2():
    start = timer()

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    # data, num_idx = dt.load_data('data/acute/acute.csv', attrs=columns, label=['label'], numerics=['a1'], pos='yes')

    columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jundice','autism',
    'used_app_before','relation']
    data, num_idx = dt.load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=['age'], pos='YES')

    # columns = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    # 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    # data, num_idx = dt.load_data('data/breastw/breastw.csv', attrs=columns, label=['label'], numerics=columns,
    # pos='malignant')

    # columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    # data, num_idx = dt.load_data('data/cars/cars.csv', attrs=columns, label=['label'], numerics=[], pos='positive')

    # columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15']
    # data, num_idx = dt.load_data('data/credit/credit.csv', attrs=columns, label=['label'],
    # numerics=['a2','a3','a8','a11','a14','a15'], pos='+')

    # columns = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    # 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina',
    # 'oldpeak', 'slope', 'major_vessels', 'thal']
    # data, num_idx = dt.load_data('data/heart/heart.csv', attrs=columns, label=['label'], numerics=['age',
    # 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak'], pos='present')

    # columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    # 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    # data, num_idx = dt.load_data('data/kidney/kidney.csv', attrs=columns, label=['label'], numerics=['age', 'bp', 'sg',
    # 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'], pos='ckd')

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    # 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    # 'a33', 'a34', 'a35', 'a36']
    # data, num_idx = dt.load_data('data/krkp/krkp.csv', attrs=columns, label=['label'], numerics=[], pos='won')

    # columns = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    # 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    # 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    # 'spore_print_color', 'population', 'habitat']
    # data, num_idx = dt.load_data('data/mushroom/mushroom.csv', attrs=columns, label=['label'], numerics=[], pos='p')

    # columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
    # 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30',
    # 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45',
    # 'a46', 'a47', 'a48', 'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    # data, num_idx = dt.load_data('data/sonar/sonar.csv', attrs=columns, label=['label'], numerics=columns, pos='Mine')

    # columns = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    # 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    # 'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    # 'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    # data, num_idx = dt.load_data('data/voting/voting.csv', attrs=columns, label=['label'], numerics=[], pos='republican')

    _, n = np.shape(data)
    res = dt.encode_data(data, num_idx, columns)

    # f = open('data/acute/file2.csv', 'w')
    f = open('data/autism/file2.csv', 'w')
    # f = open('data/breastw/file2.csv', 'w')
    # f = open('data/cars/file2.csv', 'w')
    # f = open('data/credit/file2.csv', 'w')
    # f = open('data/heart/file2.csv', 'w')
    # f = open('data/kidney/file2.csv', 'w')
    # f = open('data/krkp/file2.csv', 'w')
    # f = open('data/mushroom/file2.csv', 'w')
    # f = open('data/sonar/file2.csv', 'w')
    # f = open('data/voting/file2.csv', 'w')

    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()
    end = timer()
    print('encoding costs: ', timedelta(seconds=end - start))


def main():
    # data, attrs = sf.load_data('data/acute/file.csv')
    # data, attrs = sf.load_data('data/autism/file.csv')
    # data, attrs = sf.load_data('data/breastw/file.csv')
    # data, attrs = sf.load_data('data/cars/file.csv')
    # data, attrs = sf.load_data('data/credit/file.csv')
    # data, attrs = sf.load_data('data/heart/file.csv')
    # data, attrs = sf.load_data('data/kidney/file.csv')
    # data, attrs = sf.load_data('data/krkp/file.csv')
    # data, attrs = sf.load_data('data/mushroom/file.csv')
    # data, attrs = sf.load_data('data/sonar/file.csv')
    # data, attrs = sf.load_data('data/voting/file.csv')

    # data, attrs = sf.load_data('data/acute/file2.csv')
    data, attrs = sf.load_data('data/autism/file2.csv')
    # data, attrs = sf.load_data('data/breastw/file2.csv')
    # data, attrs = sf.load_data('data/cars/file2.csv')
    # data, attrs = sf.load_data('data/credit/file2.csv')
    # data, attrs = sf.load_data('data/heart/file2.csv')
    # data, attrs = sf.load_data('data/kidney/file2.csv')
    # data, attrs = sf.load_data('data/krkp/file2.csv')
    # data, attrs = sf.load_data('data/mushroom/file2.csv')
    # data, attrs = sf.load_data('data/sonar/file2.csv')
    # data, attrs = sf.load_data('data/voting/file2.csv')


    # data, attrs = sf.load_data('data/credit_default/file_train.csv')

    X, Y = sf.split_xy(data)
    data_train, data_test = sf.split_set(data, 0.8)
    data_train, data_valid = sf.split_set(data_train, 0.8)
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

    SHAP_pos = sf.get_shap(explainer, X_pos)
    SHAP_neg = sf.get_shap(explainer, X_neg)

    start = timer()
    depth = -1
    rules = sf.shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, depth=depth)

    fidelity, _, _, _ = sf.get_metrics(rules, X_test, Y_test_hat)
    accuracy, _, _, _ = sf.get_metrics(rules, X_test, Y_test)
    print('fidelity: ', fidelity, 'accuracy: ', accuracy)

    frules = sf.flatten_rules(rules)
    drules = sf.decode_rules(frules, attrs)
    # print(drules, len(drules))
    for r in drules:
        print(r)
    print('# of rules: ', len(drules))

    end = timer()
    print('total costs: ', timedelta(seconds=end - start))


def credit_preprocess():
    start = timer()

    columns = ['Home Ownership', 'Annual Income', 'Years in current job', 'Number of Open Accounts',
    'Years of Credit History', 'Maximum Open Credit', 'Number of Credit Problems', 'Months since last delinquent',
    'Bankruptcies', 'Purpose', 'Term', 'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 'Credit Score']

    data_train, num_idx = dt.load_data('data/credit_default/train.csv', attrs=columns, label=['Credit Default'],
    numerics=['Annual Income', 'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit',
    'Months since last delinquent', 'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 'Credit Score',],
    pos='1')

    data_test, _ = dt.load_data('data/credit_default/train.csv', attrs=columns, label=['Credit Default'],
    numerics=['Annual Income', 'Number of Open Accounts', 'Years of Credit History', 'Maximum Open Credit',
    'Months since last delinquent', 'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 'Credit Score', ],
    pos='1')

    _, n = np.shape(data_train)
    res_train, res_test = dt.encode_data2(data_train, data_test, num_idx, columns)

    f = open('data/credit_default/file_train.csv', 'w')
    for item in res_train:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()

    f = open('data/credit_default/file_test.csv', 'w')
    for item in res_test:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()

    end = timer()
    print('encoding costs: ', timedelta(seconds=end - start))


def credit_default():
    data_train, attrs = sf.load_data('data/credit_default/file_train.csv')
    data_test, _ = sf.load_data('data/credit_default/file_test.csv')

    X_train, Y_train = sf.split_xy(data_train)

    model = xgboost.XGBClassifier(objective='binary:logistic',
                                  use_label_encoder=False,
                                  n_estimators=10,
                                  max_depth=3,
                                  gamma=1,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  learning_rate=0.1
                                  ).fit(X_train, Y_train,
                                    eval_metric=["logloss"]
                                    )

    X_test, Y_test = sf.split_xy(data_test)
    Y_train_hat = model.predict(X_train)
    accuracy = accuracy_score(Y_train, Y_train_hat)
    print('xgb model accuracy on train set: ', accuracy)

    Y_train_hat = model.predict(X_train)
    explainer = shap.Explainer(model)
    X_pos, X_neg = sf.split_X_by_Y(X_train, Y_train_hat)

    SHAP_pos = sf.get_shap(explainer, X_pos)
    SHAP_neg = sf.get_shap(explainer, X_neg)

    start = timer()
    depth = -1
    rules = sf.shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, depth=depth)
    frules = sf.flatten_rules(rules)
    drules = sf.decode_rules(frules, attrs)
    print(drules, len(drules))
    end = timer()
    print('shap fold costs: ', timedelta(seconds=end - start))

    fidelity, _, _, _ = sf.get_metrics(rules, X_train, Y_train_hat)
    accuracy, _, _, _ = sf.get_metrics(rules, X_train, Y_train)
    print('fidelity: ', fidelity, 'accuracy: ', accuracy)

    print(sf.predict(rules, X_test))

    end = timer()
    print('total costs: ', timedelta(seconds=end - start))


if __name__ == '__main__':
    # preprocess()
    # preprocess2()
    # credit_preprocess()
    main()
    # credit_default()
