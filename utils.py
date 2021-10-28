import numpy as np
import random


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
    return np.array(data), attrs


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


def split_data(X, Y, ratio=0.8, rand=True):
    n = len(Y)
    k = int(n * ratio)
    train = []
    for i in range(k):
        train.append(i)
    if rand:
        for i in range(k, n):
            j = random.randint(0, i)
            if j < k:
                train[j] = i
    X_train = [X[i] for i in range(n) if i in set(train)]
    Y_train = [Y[i] for i in range(n) if i in set(train)]
    X_test = [X[i] for i in range(n) if i not in set(train)]
    Y_test = [Y[i] for i in range(n) if i not in set(train)]
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def flatten_rules(rules):
    ret = []
    rule_map = dict()
    flatten_rules.ab = -2

    def _eval(i):
        if isinstance(i, tuple):
            return _func(i)
        else:
            return i

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


def decode_rules(rules, attrs):
    ret = []

    def _f1(i):
        if i >= -1:
            s = attrs[i].split('#')
            if len(s) == 3:
                k, r, v = s[0], s[1], s[2]
                v = 'null' if len(v) == 0 else v
                if r == '==' or i == -1:
                    return k + '(X,\'' + v + '\')'
            else:
                lval, k, rval = s[0], s[2], s[4]
                _ret = k + '(X,' + 'N' + str(i) + ')'
                if lval != '-inf':
                    _ret += ',N' + str(i) + '>' + lval
                if rval != 'inf':
                    _ret += ',N' + str(i) + '<=' + rval
                return _ret
        else:
            return 'ab' + str(abs(i)) + '(X)'

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
        _ret = _ret.replace('<=', '=<')
        return _ret

    for r in rules:
        ret.append(_f2(r))
    ret.sort()
    return ret


def get_scores(Y_hat, Y):
    n = len(Y)
    if n == 0:
        return 0, 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(n):
        tp = tp + 1.0 if Y[i] and Y_hat[i] == Y[i] else tp
        tn = tn + 1.0 if not Y[i] and Y_hat[i] == Y[i] else tn
        fn = fn + 1.0 if Y[i] and Y_hat[i] != Y[i] else fn
        fp = fp + 1.0 if not Y[i] and Y_hat[i] != Y[i] else fp
    if tp < 1:
        p = 0 if fp < 1 else tp / (tp + fp)
        r = 0 if fn < 1 else tp / (tp + fn)
    else:
        p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 0 if r * p == 0 else 2 * r * p / (r + p)
    return (tp + tn) / n, p, r, f1
