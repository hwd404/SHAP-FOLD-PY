import numpy as np
import heapq
import operator


def huim_beam(X, SHAP_X, used_items=[]):
    r, c = np.shape(X)
    pool = [[]]
    values_dict, entries_dict = dict(), dict()
    width = c
    entries_dict[tuple([])] = range(r)
    visited = set()
    for i in range(c):
        done = True
        n = len(pool)
        for j in range(n):
            combo = pool[j]
            start = combo[-1] + 1 if len(combo) > 0 else i
            for k in range(start, c):
                if k in set(used_items):
                    continue
                l = np.append(combo, k)
                l = np.array(l, dtype=int)
                if tuple(l) in visited:
                    continue
                done = False
                pool.append(l)
                values_dict[tuple(l)] = float('-inf')
                entries_dict[tuple(l)] = []
                for k in entries_dict[tuple(pool[j])]:
                    if X[k][l[-1]]:
                        entries_dict[tuple(l)].append(k)
                        if values_dict[tuple(l)] == float('-inf'):
                            values_dict[tuple(l)] = 0
                        values_dict[tuple(l)] += sum([SHAP_X[k][m] for m in l])
                visited.add(tuple(l))
        if done:
            break
        top_keys = heapq.nlargest(width, values_dict, key=values_dict.get)
        pool = [list(k) for k in top_keys]
        to_remove = []
        for j in values_dict:
            if j not in set(top_keys):
                to_remove.append(j)
            elif values_dict[j] == float('-inf'):
                to_remove.append(j)
                pool.remove(list(j))
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


def predict(rules, X):
    ret = []
    for x in X:
        ret.append(classify(rules, x))
    return ret


def shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, used_items=[]):
    ret = []
    while len(X_pos) > 0:
        rule = learn_rule(X_pos, SHAP_pos, X_neg, SHAP_neg, used_items)
        tp = [i for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
        X_pos = [X_pos[i] for i in range(len(X_pos)) if i not in set(tp)]
        SHAP_pos = [SHAP_pos[i] for i in range(len(SHAP_pos)) if i not in set(tp)]
        if len(tp) == 0:
            break
        ret.append(rule)
    return ret


def learn_exception(rule, X_pos, SHAP_pos, X_neg, SHAP_neg, used_items=[]):
    SHAP_pos_ng, SHAP_neg_ng = [-x for x in SHAP_pos], [-x for x in SHAP_neg]
    ab_rules = shap_fold(X_pos, SHAP_pos_ng, X_neg, SHAP_neg_ng, used_items)
    ret = (rule[0], rule[1], ab_rules, 0)
    return ret


def learn_rule(X_pos, SHAP_pos, X_neg, SHAP_neg, used_items=[]):
    items = huim_beam(X_pos, SHAP_pos, used_items)
    rule = (-1, items, [], 0)

    fp = [i for i in range(len(X_neg)) if cover(rule, X_neg[i], 1)]
    X_fp = [X_neg[i] for i in range(len(X_neg)) if i in set(fp)]
    SHAP_fp = [SHAP_neg[i] for i in range(len(SHAP_neg)) if i in set(fp)]

    tp = [i for i in range(len(X_pos)) if cover(rule, X_pos[i], 1)]
    X_tp = [X_pos[i] for i in range(len(X_pos)) if i in set(tp)]
    SHAP_tp = [SHAP_pos[i] for i in range(len(SHAP_pos)) if i in set(tp)]

    if len(X_fp) > 0:
        rule = learn_exception(rule, X_fp, SHAP_fp, X_tp, SHAP_tp, used_items + items)
    return rule
