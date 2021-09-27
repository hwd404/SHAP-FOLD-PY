import numpy as np
import random
from timeit import default_timer as timer
from datetime import timedelta


def load_data(file, attrs=[], label=[], numerics=[], pos='', amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx, lab_idx = [], [], -1
    ret, i, k = [], 0, 0
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            attr_idx = [j for j in range(len(line)) if line[j] in attrs]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
            for j in range(len(line)):
                if line[j] in label:
                    lab_idx = j
        else:
            line = line.strip('\n').split(',')
            r = [j for j in range(len(line))]
            for j in range(len(line)):
                if j in num_idx:
                    try:
                        r[j] = float(line[j])
                    except:
                        r[j] = line[j]
                else:
                    r[j] = line[j]
            r = [r[j] for j in attr_idx]
            if lab_idx != -1:
                y = 1 if line[lab_idx] == pos else 0
                r.append(y)
            ret.append(r)
        i += 1
        amount -= 1
        if amount == 0:
            break
    n_idx = []
    i = 0
    for j in attr_idx:
        if j in num_idx:
            n_idx.append(i)
        i += 1
    random.shuffle(ret)
    return ret, n_idx


def split_set(data, ratio=0.8):
    random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


def split_xy(data):
    feature, label = [], []
    for d in data:
        feature.append(d[: -1])
        label.append(int(d[-1]))
    return feature, label


def split_X_by_Y(X, Y):
    n = len(Y)
    X_pos = [X[i] for i in range(n) if Y[i]]
    X_neg = [X[i] for i in range(n) if not Y[i]]
    return X_pos, X_neg


def evaluate(rule, x):

    def _func(i, r, v):
        if i < -1:
            return _func(-i - 2, r, v) ^ 1
        if isinstance(v, str):
            if r == '==':
                return x[i] == v
            elif r == '!=':
                return x[i] != v
            else:
                return False
        elif isinstance(x[i], str):
            return False
        elif r == '<=':
            return x[i] <= v
        elif r == '>':
            return x[i] > v
        else:
            return False

    def _eval(i):
        if len(i) == 3:
            return _func(i[0], i[1], i[2])
        elif len(i) == 4:
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


def cover(rules, x, y):
    return int(evaluate(rules, x) == y)


def classify(rules, x):
    return int(any([evaluate(r, x) for r in rules]))


def ig(tp, fn, tn, fp):
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    if tp > 0:
        ret += tp / tot * np.log(tp / tot_p)
    if fp > 0:
        ret += fp / tot * np.log(fp / tot_p)
    if tn > 0:
        ret += tn / tot * np.log(tn / tot_n)
    if fn > 0:
        ret += fn / tot * np.log(fn / tot_n)
    if tp + tn < fp + fn:
        return float('-inf')
    return ret


def best_ig(X_pos, X_neg, i, used_items=[]):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()

    for d in X_pos:
        if pos.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            pos[d[i]] += 1.0
            cp += 1.0
        else:
            xs.add(d[i])
            pos[d[i]] += 1.0
            xp += 1.0

    for d in X_neg:
        if neg.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            neg[d[i]] += 1.0
            cn += 1.0
        else:
            xs.add(d[i])
            neg[d[i]] += 1.0
            xn += 1.0

    xs = list(xs)
    xs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]

    best, v, r = float('-inf'), float('-inf'), ''

    for j in range(len(xs)):
        if (i, '<=', xs[j]) not in used_items and (i, '>', xs[j]) not in used_items:
            ifg = ig(pos[xs[j]], xp - pos[xs[j]] + cp, xn - neg[xs[j]] + cn, neg[xs[j]])
            if best < ifg:
                best, v, r = ifg, xs[j], '<='
            ifg = ig(xp - pos[xs[j]], pos[xs[j]] + cp, neg[xs[j]] + cn, xn - neg[xs[j]])
            if best < ifg:
                best, v, r = ifg, xs[j], '>'

    for c in cs:
        if (i, '==', c) not in used_items and (i, '!=', c) not in used_items:
            ifg = ig(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
            if best < ifg:
                best, v, r = ifg, c, '=='
            # ifg = ig(cp - pos[c], pos[c] + xp, neg[c] + xn, cn - neg[c])
            # if best < ifg:
            #     best, v, r = ifg, c, '!='
    return best, r, v


def best_feat(X_pos, X_neg, used_items=[]):
    if len(X_pos) == 0 and len(X_neg) == 0:
        return -1, '', ''
    n = len(X_pos[0]) if len(X_pos) > 0 else len(X_neg[0])
    _best = float('-inf')
    i, r, v = -1, '', ''
    for _i in range(n):
        bg, _r, _v = best_ig(X_pos, X_neg, _i, used_items)
        if _best < bg:
            _best = bg
            i, r, v = _i, _r, _v
    return i, r, v


def foldr(X_pos, X_neg, used_items=[]):
    ret = []
    while len(X_pos) > 0:
        rule = learn_rule(X_pos, X_neg, used_items)
        tp = [i for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_pos = [X_pos[i] for i in range(len(X_pos)) if i not in set(tp)]
        if len(tp) == 0:
            break
        ret.append(rule)
    return ret


def learn_exception(rule, X_pos, X_neg, used_items=[]):
    ab = foldr(X_pos, X_neg, used_items)
    ret = (rule[0], rule[1], ab, 0)
    return ret


def learn_rule(X_pos, X_neg, used_items=[]):
    items, added_items = [], []
    flag = False
    while True:
        t = tuple(best_feat(X_pos, X_neg, used_items + added_items))
        items.append(t)
        added_items.append(t)
        rule = (-1, items, [], 0)
        X_tp = [X_pos[i] for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_fp = [X_neg[i] for i in range(len(X_neg)) if cover(rule, X_neg[i], 1)]
        if t[0] == -1 or len(X_fp) <= 0:
            if t[0] == -1:
                items.pop()
                added_items.pop()
                rule = (-1, items, [], 0)
            if len(X_fp) > 0 and t[0] != -1:
                flag = True
            break
        X_pos = X_tp
        X_neg = X_fp
    if flag:
        rule = learn_exception(rule, X_fp, X_tp, used_items + added_items)
    return rule


def get_metrics(rules, X, Y):
    n = len(Y)
    if n == 0:
        return 0, 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
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


def acute():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data, num_idx = load_data('data/acute/acute.csv', attrs=columns, label=['label'], numerics=['a1'], pos='yes')
    return columns, data, num_idx


def autism():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
               'pdd', 'used_app_before', 'relation']
    data, num_idx = load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=['age'], pos='YES')
    return columns, data, num_idx


def breastw():
    columns = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
               'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    data, num_idx = load_data('data/breastw/breastw.csv', attrs=columns, label=['label'], numerics=columns, pos='malignant')
    return columns, data, num_idx


def cars():
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    data, num_idx = load_data('data/cars/cars.csv', attrs=columns, label=['label'], numerics=[], pos='positive')
    return columns, data, num_idx


def credit():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']
    data, num_idx = load_data('data/credit/credit.csv', attrs=columns, label=['label'],
                              numerics=['a2', 'a3', 'a8', 'a11', 'a14', 'a15'], pos='+')
    return columns, data, num_idx


def heart():
    columns = ['age', 'sex', 'chest_pain', 'blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar',
               'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina',
               'oldpeak', 'slope', 'major_vessels', 'thal']
    data, num_idx = load_data('data/heart/heart.csv', attrs=columns, label=['label'],
                              numerics=['age', 'blood_pressure', 'serum_cholestoral',
                                        'maximum_heart_rate_achieved', 'oldpeak'], pos='present')
    return columns, data, num_idx


def kidney():
    columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
               'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    data, num_idx = load_data('data/kidney/kidney.csv', attrs=columns, label=['label'],
                              numerics=['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
                                        'wbcc', 'rbcc'], pos='ckd')
    return columns, data, num_idx


def krkp():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
               'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31',
               'a32', 'a33', 'a34', 'a35', 'a36']
    data, num_idx = load_data('data/krkp/krkp.csv', attrs=columns, label=['label'], numerics=[], pos='won')
    return columns, data, num_idx


def mushroom():
    columns = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
               'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
               'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type',
               'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
    data, num_idx = load_data('data/mushroom/mushroom.csv', attrs=columns, label=['label'], numerics=[], pos='p')
    return columns, data, num_idx


def sonar():
    columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
               'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30',
               'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45',
               'a46', 'a47', 'a48', 'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60']

    data, num_idx = load_data('data/sonar/sonar.csv', attrs=columns, label=['label'], numerics=columns, pos='Mine')
    return columns, data, num_idx


def voting():
    columns = ['handicapped_infants', 'water_project_cost_sharing', 'budget_resolution', 'physician_fee_freeze',
               'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban', 'aid_to_nicaraguan_contras',
               'mx_missile', 'immigration', 'synfuels_corporation_cutback', 'education_spending',
               'superfund_right_to_sue', 'crime', 'duty_free_exports', 'export_administration_act_south_africa']
    data, num_idx = load_data('data/voting/voting.csv', attrs=columns, label=['label'], numerics=[], pos='republican')
    return columns, data, num_idx


def flatten(rules):
    ret = []
    rule_map = dict()
    flatten.ab = -2

    def _eval(i):
        if isinstance(i, tuple) and len(i) == 3:
            return i
        elif isinstance(i, tuple):
            return _func(i)

    def _func(rule, root=False):
        t = (tuple(rule[1]), tuple([_eval(i) for i in rule[2]]))
        if t not in rule_map:
            rule_map[t] = -1 if root else flatten.ab
            _ret = rule_map[t]
            ret.append((_ret, t[0], t[1]))
            if not root:
                flatten.ab -= 1
        return rule_map[t]

    for r in rules:
        _func(r, root=True)
    return ret


def decode(rules, attrs):
    ret = []

    def _f1(it):
        if isinstance(it, tuple) and len(it) == 3:
            i, r, v = it[0], it[1], it[2]
            k = attrs[i]
            v = v
            if r == '==':
                return k + '(X,' + v + ')'
            elif r == '!=':
                return 'not ' + k + '(X,' + v + ')'
            else:
                return k + '(X,' + 'N' + str(i) + ')' + ',N' + str(i) + r + str(round(v, 3))
        elif it == -1:
            return 'goal(X)'
        else:
            return'ab' + str(abs(it)) + '(X)'

    def _f2(rule):
        head = _f1(rule[0])
        body = ''
        for i in list(rule[1]):
            body = body + _f1(i) + ','
        tail = ''
        for i in list(rule[2]):
            t = _f1(i)
            if 'not' not in t:
                tail = tail + 'not ' + _f1(i) + ','
            else:
                t = t.replace('not ', '')
                tail = tail + t + ','
        _ret = head + ':-' + body + tail
        chars = list(_ret)
        chars[-1] = '.'
        _ret = ''.join(chars)
        return _ret

    for r in rules:
        ret.append(_f2(r))
    ret.sort()
    return ret


def main():
    # columns, data, num_idx = acute()
    # columns, data, num_idx = autism()
    # columns, data, num_idx = breastw()
    # columns, data, num_idx = cars()
    # columns, data, num_idx = credit()
    # columns, data, num_idx = heart()
    # columns, data, num_idx = kidney()
    # columns, data, num_idx = krkp()
    # columns, data, num_idx = mushroom()
    columns, data, num_idx = sonar()
    # columns, data, num_idx = voting()


    m, n = np.shape(data)
    data_train, data_test = split_set(data, 0.8)
    print('total # of data ', m, n)
    X, Y = split_xy(data_train)
    X_pos, X_neg = split_X_by_Y(X, Y)
    X, Y = split_xy(data_test)

    start = timer()
    rules1 = foldr(X_pos, X_neg, [])
    fr1 = flatten(rules1)
    print(decode(fr1, columns), len(fr1))
    acc, _, _, _ = get_metrics(rules1, X, Y)
    print('accuracy', round(acc, 4))

    end = timer()
    print('fold costs: ', timedelta(seconds=end - start))


if __name__ == '__main__':
    main()
