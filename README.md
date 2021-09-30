# SHAP-FOLD-PY
A Python implementation of SHAP-FOLD algorithms.\
The author of SHAP-FOLD also has an original implementation: \
https://github.com/fxs130430/SHAP_FOLD \
You can find the research paper and description in this repository.
<!--A novel contribution in this Python implementation is that the High Utility Itemset Mining has been built with Beam search and some optimization, which provides a decent performance for HUIM.--> 

## Install
### Prerequisites
SHAP-FOLD-PY is developed with only python3. Here are the dependencies:
* SHAP <pre> python3 -m pip install shap </pre>
* XGBoost <pre> python3 -m pip install xgboost </pre>
* sklearn <pre> python3 -m pip install scikit-learn </pre>

If there are still missing libraries, you can use the same command to install them.

## Instruction
### Data preparation
The SHAP-FOLD algorithm takes binary tabular data as input, each column should be an independent binary feature. \
Numeric features would be mapped to as few numerical intervals as possible. \
We have prepared two different encoders in this repo for tabular data: one-hot encoder and decision tree encoder. 
+ one-hot encoder can be used for simple / small datasets.
+ decision tree encoder can be used for larger datasets. Only the features with high information gain would be selected.
   
For example, the UCI acute dataset can be encoded to 46 features with one-hot encoding but only encoded to 12 features with decision tree encoding.\
Here is an example:

<code>

    import decision_tree_encoding as dt
    columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jundice','autism', 'used_app_before','relation']
    data_train, num_idx = dt.load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=['age'], pos='YES')
</code>

**columns** lists all the features needed, **numerics** lists all the numeric features, **label** implies the feature name of the label, **pos** indicates the positive value of the label.

### Training
The SHAP-FOLD algorithm is to generate an explainable model that are represented with an answer set program to explain an existing classification model. Here's an example:

<code>

    import shap_fold as sf 
    X_train, Y_train = sf.split_xy(data_train)
    model = xgboost.XGBClassifier(objective='binary:logistic').fit(X_train, Y_train, eval_metric=["logloss"])
</code>

We got a xgboost model with above code, then: 

<code>

    Y_train_hat = model.predict(X_train)
    explainer = shap.Explainer(model)
    X_pos, X_neg = sf.split_X_by_Y(X_train, Y_train_hat)

    SHAP_pos = sf.get_shap(explainer, X_pos)
    SHAP_neg = sf.get_shap(explainer, X_neg)
</code>

SHAP_pos, SHAP_neg are the shapley value matrix for positive data and negative data.

<code>

    rules = sf.shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg)
    frules = sf.flatten_rules(rules)
    drules = sf.decode_rules(frules, attrs)
</code>

**drules** are the rule set as result. \
There are many UCI dataset in this repo as examples, you can find the details in the code files.

### Limits

The recommended number of feature columns should be less than 1500. The time consumption are much more sensitive to the number of columns. \
The time complexity is roughly polynominal to the number of instances and exponential to the number of features for both shapley value caculation and HUIM. \
A tabular dataset with 200 rows and 1500 columns would take about 50 minutes to finish on a desktop with 6 core i5 cpu and 32 GB memory.
<!--
### FOLD-R

SHAP-FOLD has some limitations on scalability. Computational work load would increase while the number of rows or columns increased, especially on columns. \
In this scenario, FOLD-R can be used instead. But, the SHAP-FOLD is still better on standard metrics and explainability.\
The FOLD-R algorithm can be applied on the original tabular data file without encoding. The usage is similar to the above process, the details is in the foldr.py.
--> 
