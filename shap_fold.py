import xgboost
import shap
import numpy as np
import operator
import random
import heapq
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from datetime import timedelta


def load_data(data_file, amount=-1):
    f = open(data_file, 'r')
    num = 0
    data, attrs = [], []
    for line in f.readlines():
        if num < 1:
            attrs = line.strip('\n').split(',')
            num += 1
            continue
        line = line.strip('\n').split(',')
        line = [int(i) for i in line]
        data.append(line)
        if amount == 0:
            break
        amount = -1
    random.shuffle(data)
    return np.array(data), attrs


def split_set(data, ratio=0.8):
    random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


def split_cls(data):
    pos_data = [d for d in data if d[-1] > 0]
    neg_data = [d for d in data if d[-1] < 1]
    return pos_data, neg_data


def split_xy(data):
    feature, label = [], []
    for d in data:
        feature.append(d[: -1])
        label.append(int(d[-1]))
    return np.array(feature), np.array(label)


def split_X_by_Y(X, Y):
    n = len(Y)
    X_pos = [X[i] for i in range(n) if Y[i]]
    X_neg = [X[i] for i in range(n) if not Y[i]]
    return np.array(X_pos), np.array(X_neg)


def get_shap(explainer, X):
    return np.array(explainer(X).values)


# --------------------------------------------------------------------------------


def huim_beam(X, shap_values, used_items=[]):
    r, c = np.shape(X)
    cands = [[]]
    values_dict, entries_dict = dict(), dict()
    width = c
    entries_dict[tuple([])] = range(r)
    visited = set()
    for i in range(c):
        done = True
        n = len(cands)
        for j in range(n):
            cand = cands[j]
            start = cand[-1] + 1 if len(cand) > 0 else i
            for k in range(start, c):
                if k in set(used_items):
                    continue
                l = np.append(cand, k)
                l = np.array(l, dtype=int)
                if tuple(l) in visited:
                    continue
                done = False
                cands.append(l)
                values_dict[tuple(l)] = float('-inf')
                entries_dict[tuple(l)] = []
                for k in entries_dict[tuple(cands[j])]:
                    if X[k][l[-1]]:
                        entries_dict[tuple(l)].append(k)
                        if values_dict[tuple(l)] == float('-inf'):
                            values_dict[tuple(l)] = 0
                        values_dict[tuple(l)] += sum([shap_values[k][m] for m in l])
                visited.add(tuple(l))
        if done:
            break
        top_keys = heapq.nlargest(width, values_dict, key=values_dict.get)
        cands = [list(k) for k in top_keys]
        to_remove = []
        for j in values_dict:
            if j not in set(top_keys):
                to_remove.append(j)
            elif values_dict[j] == float('-inf'):
                to_remove.append(j)
                cands.remove(list(j))
        for j in to_remove:
            values_dict.pop(j)
            entries_dict.pop(j)

    ret = max(values_dict.items(), key=operator.itemgetter(1))[0] if len(values_dict.items()) > 0 else []
    return list(ret)


def evaluate(rule, x):

    def _neg(i):
        return -i - 2

    def _eval(i):
        if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
            if i >= 0:
                return x[i]
            else:
                return x[_neg(i)] ^ 1
        elif isinstance(i, tuple):
            return evaluate(i, x)

    if len(rule) == 0:
        return 0
    if rule[3] == 0 and not all([_eval(i) for i in rule[1]]):
        return 0
    if rule[3] == 1 and not any([_eval(i) for i in rule[1]]):
        return 0
    if len(rule[2]) > 0 and any([_eval(i) for i in rule[2]]):
        return 0
    return 1


def cover(rule, x, y):
    return int(evaluate(rule, x) == y)


def classify(rules, x):
    return int(any([evaluate(r, x) for r in rules]))


def flatten_rules(rules):
    ret = []
    rule_map = dict()
    flatten_rules.ab = -2

    def _eval(i):
        if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
            return i
        elif isinstance(i, tuple):
            return _func(i)

    def _func(rule, root=False):
        t = (tuple(rule[1]), tuple([_eval(i) for i in rule[2]]))
        if t not in rule_map:
            rule_map[t] = -1 if root else flatten_rules.ab
            _ret = rule_map[t]
            ret.append((_ret, t[0], t[1]))
            if not root:
                flatten_rules.ab -= 1
        return rule_map[t]

    for r in rules:
        _func(r, root=True)
    return ret


def decode_rules(rules, attrs, numerics=[]):
    ret = []

    def _f1(i):
        if i >= 0:
            strs = attrs[i].split('_')
            v = strs[-1]
            k = '_'.join(strs[:-1])
            strs = k.split(' ')
            k = '_'.join(strs)
            v = str(v).replace(' ', '_')
            v = 'null' if len(v) == 0 else v
            return k + '(X,' + v + ')'
        elif i == -1:
            return 'goal(X)'
        else:
            return'ab' + str(abs(i)) + '(X)'

    def _f2(rule):
        head = _f1(rule[0])
        body = ''
        for i in list(rule[1]):
            body = body + _f1(i) + ','
        tail = ''
        for i in list(rule[2]):
            tail = tail + 'not ' + _f1(i) + ','
        _ret = head + ':-' + body + tail
        chars = list(_ret)
        chars[-1] = '.'
        _ret = ''.join(chars)
        return _ret

    for r in rules:
        ret.append(_f2(r))
    ret.sort()
    return ret


def shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, depth=-1, used_items=[]):
    ret = []
    while len(X_pos) > 0 and depth != 0:
        rule = learn_rule(X_pos, SHAP_pos, X_neg, SHAP_neg, depth, used_items)
        tp = [i for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_pos = [X_pos[i] for i in range(len(X_pos)) if i not in set(tp)]
        SHAP_pos = [SHAP_pos[i] for i in range(len(SHAP_pos)) if i not in set(tp)]
        if len(tp) == 0:
            break
        ret.append(rule)
    return ret


def learn_exception(rule, X_pos, SHAP_pos, X_neg, SHAP_neg, depth, used_items=[]):
    SHAP_pos_ng, SHAP_neg_ng = [-x for x in SHAP_pos], [-x for x in SHAP_neg]
    ab_rules = shap_fold(X_pos, SHAP_pos_ng, X_neg, SHAP_neg_ng, depth, used_items)
    if len(X_pos) > 0 and len(ab_rules) == 0:
        print('failed to generate exception for rule ', rule, ' when # of pos is ', len(X_pos), ' # of neg is ', len(X_neg))

    ret = (rule[0], rule[1], ab_rules, 0)
    return ret


def learn_rule(X_pos, SHAP_pos, X_neg, SHAP_neg, depth, used_items=[]):
    items = huim_beam(X_pos, SHAP_pos, used_items)
    ret = (-1, items, [], 0)
    if len(items) == 0:
        print('can not learn any rule due to no item set')

    fp = [i for i in range(len(X_neg)) if cover(ret, X_neg[i], 1)]
    X_fp = [X_neg[i] for i in range(len(X_neg)) if i in set(fp)]
    SHAP_fp = [SHAP_neg[i] for i in range(len(SHAP_neg)) if i in set(fp)]

    tp = [i for i in range(len(X_pos)) if cover(ret, X_pos[i], 1)]
    X_tp = [X_pos[i] for i in range(len(X_pos)) if i in set(tp)]
    SHAP_tp = [SHAP_pos[i] for i in range(len(SHAP_pos)) if i in set(tp)]

    ret = learn_exception(ret, X_fp, SHAP_fp, X_tp, SHAP_tp, depth - 1, used_items + items)
    return ret


def flatten(rule):
    def _mul(a, b):
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        _ret = []
        for i in a:
            for j in b:
                if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
                    i = [i]
                if isinstance(j, int) or isinstance(j, np.int32) or isinstance(j, np.int64):
                    j = [j]
                k = []
                for _i in i:
                    if _i not in k:
                        k.append(_i)
                for _j in j:
                    if _j not in k:
                        k.append(_j)
                k.sort()
                if k not in _ret:
                    _ret.append(k)
        return _ret

    def _f(i):
        if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):
            return [i]
        if isinstance(i, tuple):
            return _dfs(i)

    def _dfs(_rule):
        _ret = []
        if _rule[3] == 1:
            for i in _rule[1]:
                if isinstance(i, tuple):
                    for j in _f(i):
                        _ret.append(j)
                else:
                    _ret.append(_f(i))
            return _ret
        else:
            _ret = [[]]
            for i in _rule[1]:
                _ret = _mul(_ret, _f(i))
        return _ret
    rules = _dfs(rule)
    ret = []
    for r in rules:
        ret.append((-1, r, [], 0))
    return ret


def get_metrics(rules, X, Y):
    n = len(Y)
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        tp = tp + 1.0 if Y[i] and classify(rules, X[i]) == Y[i] else tp
        tn = tn + 1.0 if not Y[i] and classify(rules, X[i]) == Y[i] else tn
        fn = fn + 1.0 if Y[i] and classify(rules, X[i]) != Y[i] else fn
        fp = fp + 1.0 if not Y[i] and classify(rules, X[i]) != Y[i] else fp
    if tp < 1:
        p = 0 if fp < 1 else tp / (tp + fp)
        r = 0 if fn < 1 else tp / (tp + fn)
    else:
        p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 0 if r * p == 0 else 2 * r * p / (r + p)
    return (tp + tn) / n, p, r, f1


def predict(rules, X):
    ret = []
    for x in X:
        ret.append(classify(rules, x))
    return ret


def main():
    data, attrs = load_data('data/autism/file2.csv')

    X, Y = split_xy(data)
    data_train, data_test = split_set(data, 0.8)
    data_train, data_valid = split_set(data_train, 0.5)
    X_train, Y_train = split_xy(data_train)
    X_valid, Y_valid = split_xy(data_valid)
    valid_set = [(X_valid, Y_valid)]
    model = xgboost.XGBClassifier(objective='binary:logistic',
                                  use_label_encoder=False,
                                  n_estimators=10,
                                  max_depth=3,
                                  gamma=1,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  learning_rate=0.1
                                  ).fit(X_train, Y_train, early_stopping_rounds=100, eval_set=valid_set, eval_metric=["logloss"])

    X_test, Y_test = split_xy(data_test)
    Y_test_hat = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_test_hat)
    print('xgb model accuracy: ', accuracy)

    Y_train_hat = model.predict(X_train)
    explainer = shap.Explainer(model)
    X_pos, X_neg = split_X_by_Y(X_train, Y_train_hat)

    SHAP_pos = get_shap(explainer, X_pos)
    SHAP_neg = get_shap(explainer, X_neg)

    start = timer()
    depth = -1
    rules = shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, depth=depth)

    print(rules, len(rules))
    fidelity, _, _, _ = get_metrics(rules, X_test, Y_test_hat)
    accuracy, _, _, _ = get_metrics(rules, X_test, Y_test)
    print('fidelity: ', fidelity, 'accuracy: ', accuracy)

    frules = flatten_rules(rules)
    drules = decode_rules(frules, attrs)
    print(drules, len(drules))

    end = timer()
    print('total costs: ', timedelta(seconds=end - start))


if __name__ == '__main__':
    main()
