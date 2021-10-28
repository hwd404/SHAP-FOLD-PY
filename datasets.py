from encoder import TreeEncoder, OneHotEncoder, save_data_to_file


def encode_acute():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    nums = ['a1']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='yes')
    data = encoder.encode('data/acute/acute.csv')
    save_data_to_file(data, 'data/acute/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='yes')
    data = encoder.encode('data/acute/acute.csv')
    save_data_to_file(data, 'data/acute/file2.csv')


def encode_autism():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
             'autism', 'used_app_before', 'relation']
    nums = ['age']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='NO')
    data = encoder.encode('data/autism/autism.csv')
    save_data_to_file(data, 'data/autism/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='NO')
    data = encoder.encode('data/autism/autism.csv')
    save_data_to_file(data, 'data/autism/file2.csv')


def encode_breastw():
    attrs = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
    'single_epi_cell_size', 'bare_nuclei', 'bland_chrodatain', 'normal_nucleoli', 'mitoses']
    nums = attrs

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='benign')
    data = encoder.encode('data/breastw/breastw.csv')
    save_data_to_file(data, 'data/breastw/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='benign')
    data = encoder.encode('data/breastw/breastw.csv')
    save_data_to_file(data, 'data/breastw/file2.csv')


def encode_cars():
    attrs = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    nums = attrs

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='negative')
    data = encoder.encode('data/cars/cars.csv')
    save_data_to_file(data, 'data/cars/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='negative')
    data = encoder.encode('data/cars/cars.csv')
    save_data_to_file(data, 'data/cars/file2.csv')


def encode_credit():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    nums = ['a2', 'a3', 'a8', 'a11', 'a14', 'a15']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='-')
    data = encoder.encode('data/credit/credit.csv')
    save_data_to_file(data, 'data/credit/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='-')
    data = encoder.encode('data/credit/credit.csv')
    save_data_to_file(data, 'data/credit/file2.csv')


def encode_heart():
    attrs = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
    'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
    'slope', 'major_vessels', 'thal']
    nums = ['age', 'blood_pressure', 'serum_cholestoral', 'maximum_heart_rate_achieved', 'oldpeak']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='absent')
    data = encoder.encode('data/heart/heart.csv')
    save_data_to_file(data, 'data/heart/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='absent')
    data = encoder.encode('data/heart/heart.csv')
    save_data_to_file(data, 'data/heart/file2.csv')


def encode_kidney():
    attrs = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    nums = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='ckd')
    data = encoder.encode('data/kidney/kidney.csv')
    save_data_to_file(data, 'data/kidney/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='ckd')
    data = encoder.encode('data/kidney/kidney.csv')
    save_data_to_file(data, 'data/kidney/file2.csv')


def encode_krkp():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36']
    nums = []

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='won')
    data = encoder.encode('data/krkp/krkp.csv')
    save_data_to_file(data, 'data/krkp/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='won')
    data = encoder.encode('data/krkp/krkp.csv')
    save_data_to_file(data, 'data/krkp/file2.csv')


def encode_mushroom():
    attrs = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',
    'spore_print_color', 'population', 'habitat']
    nums = []

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='p')
    data = encoder.encode('data/mushroom/mushroom.csv')
    save_data_to_file(data, 'data/mushroom/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='p')
    data = encoder.encode('data/mushroom/mushroom.csv')
    save_data_to_file(data, 'data/mushroom/file2.csv')


def encode_sonar():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
    'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32',
    'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48',
    'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']
    nums = attrs

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='Mine')
    data = encoder.encode('data/sonar/sonar.csv')
    save_data_to_file(data, 'data/sonar/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='Mine')
    data = encoder.encode('data/sonar/sonar.csv')
    save_data_to_file(data, 'data/sonar/file2.csv')


def encode_voting():
    attrs = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
    'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
    'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending', 'superfund_right_to_sue',
    'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    nums = []

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='republican')
    data = encoder.encode('data/voting/voting.csv')
    save_data_to_file(data, 'data/voting/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='republican')
    data = encoder.encode('data/voting/voting.csv')
    save_data_to_file(data, 'data/voting/file2.csv')


def encode_ecoli():
    attrs = ['sn','mcg','gvh','lip','chg','aac','alm1','alm2']
    nums = ['mcg','gvh','lip','chg','aac','alm1','alm2']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='cp')
    data = encoder.encode('data/ecoli/ecoli.csv')
    save_data_to_file(data, 'data/ecoli/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='cp')
    data = encoder.encode('data/ecoli/ecoli.csv')
    save_data_to_file(data, 'data/ecoli/file2.csv')


def encode_ionosphere():
    attrs = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19',
    'c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34']
    nums = attrs

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='g')
    data = encoder.encode('data/ionosphere/ionosphere.csv')
    save_data_to_file(data, 'data/ionosphere/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='g')
    data = encoder.encode('data/ionosphere/ionosphere.csv')
    save_data_to_file(data, 'data/ionosphere/file2.csv')


def encode_wine():
    attrs = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','tot_phenols','flavanoids',
    'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','OD_of_diluted','proline']
    nums = attrs

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='3')
    data = encoder.encode('data/wine/wine.csv')
    save_data_to_file(data, 'data/wine/file1.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='3')
    data = encoder.encode('data/wine/wine.csv')
    save_data_to_file(data, 'data/wine/file2.csv')


def encode_adult():
    attrs = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship',
    'race','sex','capital_gain','capital_loss','hours_per_week','native_country']
    nums = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='<=50K')
    data = encoder.encode('data/adult/adult.csv')
    save_data_to_file(data, 'data/adult/file2.csv')


def encode_credit_card():
    attrs = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
    'PAY_AMT5','PAY_AMT6']
    nums = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
    'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='DEFAULT_PAYMENT', pos='0')
    data = encoder.encode('data/credit_card/credit_card.csv')
    save_data_to_file(data, 'data/credit_card/file2.csv')


def encode_titanic():
    attrs = ['Sex', 'Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare', 'Class', 'Embarked']
    nums = ['Age', 'Number_of_Siblings_Spouses', 'Number_Of_Parents_Children', 'Fare']

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='Survived', pos='0')
    data = encoder.encode('data/titanic/train.csv')
    save_data_to_file(data, 'data/titanic/file_train.csv')
    data = encoder.encode('data/titanic/test.csv')
    save_data_to_file(data, 'data/titanic/file_test.csv')


def main():
    encode_acute()
    encode_autism()
    encode_breastw()
    encode_cars()
    encode_credit()
    encode_ecoli()
    encode_heart()
    encode_ionosphere()
    encode_kidney()
    encode_krkp()
    encode_mushroom()
    encode_sonar()
    encode_voting()
    encode_wine()
    # encode_adult()
    # encode_credit_card()
    encode_titanic()


if __name__ == '__main__':
    main()
