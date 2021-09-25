'''
This file encodes tabular data using decision tree with c4.5 method:
only the features with best information gain would be selected.
'''
import numpy as np


def load_data(file, attrs=[], label=[], numerics=[], pos='', amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx, lab_idx = [], [], -1
    ret, i = [], 0
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
        if amount == 0:
            break
        amount -= 1
    n_idx = []
    i = 0
    for j in attr_idx:
        if j in num_idx:
            n_idx.append(i)
        i += 1
    return ret, n_idx


def entropy(data, idx=-1):
    ret = 0.0
    if len(data) == 0:
        return ret
    n, _ = np.shape(data)
    ys = dict()
    for d in data:
        if ys.get(d[idx]) is None:
            ys[d[idx]] = 0.0
        ys[d[idx]] += 1.0
    for y in ys:
        ys[y] /= n
        if ys[y]:
            ret -= ys[y] * np.log(ys[y])
    return ret


def info_gain(data, idx):
    m, _ = np.shape(data)
    ret = 0.0
    pos, neg = dict(), dict()
    for d in data:
        if pos.get(d[idx]) is None:
            pos[d[idx]], neg[d[idx]] = 0.0, 0.0
        if d[-1] > 0.0:
            pos[d[idx]] += 1.0
        else:
            neg[d[idx]] += 1.0
    for i in pos:
        tot = pos[i] + neg[i]
        pos[i] /= tot
        neg[i] /= tot
        if pos[i] > 0:
            ret -= (tot / m) * pos[i] * np.log(pos[i])
        if neg[i] > 0:
            ret -= (tot / m) * neg[i] * np.log(neg[i])
    return entropy(data) - ret


def info_gain_numeric(data, idx, pivot):
    m, _ = np.shape(data)
    ret = 0.0
    pos, neg = dict(), dict()
    pos[-1], neg[-1] = 0.0, 0.0
    pos[1], neg[1] = 0.0, 0.0
    for d in data:
        try:
            key = -1 if d[idx] <= pivot else 1
        except:
            key = d[idx]
            if pos.get(key) is None:
                pos[key], neg[key] = 0.0, 0.0
        if d[-1] > 0.0:
            pos[key] += 1.0
        else:
            neg[key] += 1.0
    for i in pos:
        tot = pos[i] + neg[i]
        pos[i] /= tot
        neg[i] /= tot
        if pos[i] > 0.0:
            ret -= (tot / m) * pos[i] * np.log(pos[i])
        if neg[i] > 0.0:
            ret -= (tot / m) * neg[i] * np.log(neg[i])
    return entropy(data) - ret


def best_info_gain2(data, idx):
    m, _ = np.shape(data)
    xs = set()
    for d in data:
        try:
            xs.add(float(d[idx]))
        except:
            pass
    xs = list(xs)
    xs.sort()
    best_ig, split = float('-inf'), float('-inf')
    for i in range(len(xs) - 1):
        pivot = (xs[i] + xs[i + 1]) / 2
        ig = info_gain_numeric(data, idx, pivot)
        if best_ig < ig:
            best_ig = ig
            split = pivot
    return best_ig, split


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
    return ret


def best_info_gain(data, idx):
    m, _ = np.shape(data)
    p, n = 0, 0
    pos, neg = dict(), dict()
    xs = set()

    for d in data:
        try:
            key = float(d[idx])
            xs.add(key)
            if pos.get(key) is None:
                pos[key], neg[key] = 0, 0
            if d[-1] > 0:
                pos[key] += 1.0
                p += 1.0
            else:
                neg[key] += 1.0
                n += 1.0
        except:
            pass
    xs = list(xs)
    xs.sort()

    for i in range(1, len(xs)):
        pos[xs[i]] += pos[xs[i - 1]]
        neg[xs[i]] += neg[xs[i - 1]]

    best_ig, split = float('-inf'), float('-inf')

    for i in range(len(xs)):
        ifg = ig(p - pos[xs[i]], pos[xs[i]], neg[xs[i]], n - neg[xs[i]])
        if best_ig < ifg:
            best_ig = ifg
            split = xs[i]

    return entropy(data) + best_ig, split


class Node(object):
    def __init__(self, data, num_idx=[]):
        self.data = data
        self.index = -1
        self.pivot = -1
        self.extras = set()
        self.children = dict()
        self.num_idx = num_idx


def split_node(node):
    data = node.data
    if len(data) == 0 or entropy(data) == 0.0:
        return
    m, n = np.shape(data)
    max_ig = float('-inf')
    max_idx = -1
    for i in range(0, n - 1):
        if i in node.num_idx:
            ig, _ = best_info_gain(data, i)
        else:
            ig = info_gain(data,i)
        if max_ig < ig:
            max_ig, max_idx = ig, i
    if max_ig == 0:
        return
    node.index = max_idx
    if max_idx in node.num_idx:
        _, node.pivot = best_info_gain(data, max_idx)
        buckets = dict()
        left_name, right_name = '<=', '>'
        left_data, right_data = [], []
        for d in data:
            try:
                if d[max_idx] <= node.pivot:
                    left_data.append(d)
                else:
                    right_data.append(d)
            except:
                key = d[max_idx]
                if buckets.get(key) is None:
                    buckets[key] = []
                node.extras.add(key)
                buckets[key].append(d)
        if len(left_data) == 0 or len(right_data) == 0:
            return
        node.children[left_name] = Node(left_data, node.num_idx)
        node.children[right_name] = Node(right_data, node.num_idx)
        for i in buckets:
            node.children[i] = Node(buckets[i], node.num_idx)
            node.children[i].pivot = i
    else:
        buckets = dict()
        for d in data:
            if buckets.get(d[max_idx]) is None:
                buckets[d[max_idx]] = []
            buckets[d[max_idx]].append(d)
        for i in buckets:
            node.children[i] = Node(buckets[i], node.num_idx)
    for i in node.children:
        split_node(node.children[i])


def build_tree(data, num_idx):
    ret = Node(data, num_idx)
    split_node(ret)
    return ret


def traverse(node, path, splits=dict()):
    if node is None:
        pass
    if entropy(node.data) == 0.0:
        pass
    for i in node.children:
        path.append([node.index, node.pivot, node.extras])
        if splits.get(node.index) is None:
            splits[node.index] = set()
        if node.index in node.num_idx:
            splits[node.index].add(node.pivot)
            for e in node.extras:
                splits[node.index].add(e)
        else:
            splits[node.index].add(i)
        traverse(node.children[i], path, splits)
        path.pop()


def encode_data(data, num_idx, columns=[]):
    m, n = np.shape(data)
    attr_splits = dict()

    root = build_tree(data, num_idx)
    traverse(root, [], attr_splits)
    for d in data:
        for i in range(n - 1):
            if i in num_idx:
                if isinstance(d[i], str):
                    if attr_splits.get(i) is None:
                        attr_splits[i] = set()
                    attr_splits[i].add(d[i])
            else:
                if attr_splits.get(i) is None:
                    attr_splits[i] = set()
                attr_splits[i].add(d[i])

    attrs, k = [], 0
    attrs_map = dict()
    for i in attr_splits:
        if i in num_idx:
            values, extras = [], set()
            for j in attr_splits[i]:
                try:
                    values.append(float(j))
                except:
                    extras.add(j)
            if len(values) == 0:
                continue
            values.sort()
            for j in range(len(values)):
                if j == 0:
                    name = columns[i] + '_' + '-inf' + '-' + str(round(values[j], 3))
                else:
                    name = columns[i] + '_' + str(round(values[j - 1], 3)) + '-' + str(round(values[j], 3))
                attrs.append(name)
                attrs_map[name] = k
                k += 1
            name = columns[i] + '_' + str(round(values[-1], 3)) + '-' + 'inf'
            attrs.append(name)
            attrs_map[name] = k
            k += 1
            for e in extras:
                name = columns[i] + '_' + e
                attrs.append(name)
                attrs_map[name] = k
                k += 1
        else:
            values = attr_splits[i]
            for v in values:
                name = columns[i] + '_' + str(v)
                attrs.append(name)
                attrs_map[name] = k
                k += 1

    attrs.append('label')
    ret = [attrs]
    for d in data:
        row = [0] * k
        for i in range(n - 1):
            if i in num_idx:
                if i not in attr_splits:
                    continue
                values = []
                for j in attr_splits[i]:
                    try:
                        values.append(float(j))
                    except:
                        pass
                values.sort()
                for j in range(len(values)):
                    if j == 0:
                        l, r = float('-inf'), values[j]
                    else:
                        l, r = values[j - 1], values[j]
                    try:
                        if l < d[i] <= r:
                            name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                            row[attrs_map[name]] = 1
                    except:
                        name = columns[i] + '_' + d[i]
                        row[attrs_map[name]] = 1
                l, r = values[-1] if len(values) > 0 else float('-inf'), float('inf')
                try:
                    if l < d[i] <= r:
                        name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                        row[attrs_map[name]] = 1
                except:
                    pass
            else:
                name = columns[i] + '_' + d[i]
                row[attrs_map[name]] = 1
        row.append(d[-1])
        ret.append(row)
    return ret


# for unlabeled test data
def encode_data2(data_train, data_test, num_idx, columns=[]):
    m, n = np.shape(data_train)
    attr_splits = dict()

    root = build_tree(data_train, num_idx)
    traverse(root, [], attr_splits)
    for d in data_train:
        for i in range(n - 1):
            if i in num_idx:
                if isinstance(d[i], str):
                    if attr_splits.get(i) is None:
                        attr_splits[i] = set()
                    attr_splits[i].add(d[i])
            else:
                if attr_splits.get(i) is None:
                    attr_splits[i] = set()
                attr_splits[i].add(d[i])

    attrs, k = [], 0
    attrs_map = dict()
    for i in attr_splits:
        if i in num_idx:
            values, extras = [], set()
            for j in attr_splits[i]:
                try:
                    values.append(float(j))
                except:
                    extras.add(j)
            if len(values) == 0:
                continue
            values.sort()
            for j in range(len(values)):
                if j == 0:
                    name = columns[i] + '_' + '-inf' + '-' + str(round(values[j], 3))
                else:
                    name = columns[i] + '_' + str(round(values[j - 1], 3)) + '-' + str(round(values[j], 3))
                attrs.append(name)
                attrs_map[name] = k
                k += 1
            name = columns[i] + '_' + str(round(values[-1], 3)) + '-' + 'inf'
            attrs.append(name)
            attrs_map[name] = k
            k += 1
            for e in extras:
                name = columns[i] + '_' + e
                attrs.append(name)
                attrs_map[name] = k
                k += 1
        else:
            values = attr_splits[i]
            for v in values:
                name = columns[i] + '_' + str(v)
                attrs.append(name)
                attrs_map[name] = k
                k += 1

    attrs.append('label')
    ret_train, ret_test = [attrs], [attrs]
    for d in data_train:
        row = [0] * k
        for i in range(n - 1):
            if i in num_idx:
                if i not in attr_splits:
                    continue
                values = []
                for j in attr_splits[i]:
                    try:
                        values.append(float(j))
                    except:
                        pass
                values.sort()
                for j in range(len(values)):
                    if j == 0:
                        l, r = float('-inf'), values[j]
                    else:
                        l, r = values[j - 1], values[j]
                    try:
                        if l < d[i] <= r:
                            name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                            row[attrs_map[name]] = 1
                    except:
                        name = columns[i] + '_' + d[i]
                        row[attrs_map[name]] = 1
                l, r = values[-1] if len(values) > 0 else float('-inf'), float('inf')
                try:
                    if l < d[i] <= r:
                        name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                        row[attrs_map[name]] = 1
                except:
                    pass
            else:
                name = columns[i] + '_' + d[i]
                row[attrs_map[name]] = 1
        row.append(d[-1])
        ret_train.append(row)

    for d in data_test:
        row = [0] * k
        for i in range(n - 1):
            if i in num_idx:
                if i not in attr_splits:
                    continue
                values = []
                for j in attr_splits[i]:
                    try:
                        values.append(float(j))
                    except:
                        pass
                values.sort()
                for j in range(len(values)):
                    if j == 0:
                        l, r = float('-inf'), values[j]
                    else:
                        l, r = values[j - 1], values[j]
                    try:
                        if l < d[i] <= r:
                            name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                            row[attrs_map[name]] = 1
                    except:
                        name = columns[i] + '_' + d[i]
                        row[attrs_map[name]] = 1
                l, r = values[-1] if len(values) > 0 else float('-inf'), float('inf')
                try:
                    if l < d[i] <= r:
                        name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                        row[attrs_map[name]] = 1
                except:
                    pass
            else:
                name = columns[i] + '_' + d[i]
                row[attrs_map[name]] = 1
        row.append(0)
        ret_test.append(row)

    return ret_train, ret_test


# for labeled test data
def encode_data3(data_train, data_test, num_idx, columns=[]):
    m, n = np.shape(data_train)
    attr_splits = dict()

    root = build_tree(data_train, num_idx)
    traverse(root, [], attr_splits)
    for d in data_train:
        for i in range(n - 1):
            if i in num_idx:
                if isinstance(d[i], str):
                    if attr_splits.get(i) is None:
                        attr_splits[i] = set()
                    attr_splits[i].add(d[i])
            else:
                if attr_splits.get(i) is None:
                    attr_splits[i] = set()
                attr_splits[i].add(d[i])

    attrs, k = [], 0
    attrs_map = dict()
    for i in attr_splits:
        if i in num_idx:
            values, extras = [], set()
            for j in attr_splits[i]:
                try:
                    values.append(float(j))
                except:
                    extras.add(j)
            if len(values) == 0:
                continue
            values.sort()
            for j in range(len(values)):
                if j == 0:
                    name = columns[i] + '_' + '-inf' + '-' + str(round(values[j], 3))
                else:
                    name = columns[i] + '_' + str(round(values[j - 1], 3)) + '-' + str(round(values[j], 3))
                attrs.append(name)
                attrs_map[name] = k
                k += 1
            name = columns[i] + '_' + str(round(values[-1], 3)) + '-' + 'inf'
            attrs.append(name)
            attrs_map[name] = k
            k += 1
            for e in extras:
                name = columns[i] + '_' + e
                attrs.append(name)
                attrs_map[name] = k
                k += 1
        else:
            values = attr_splits[i]
            for v in values:
                name = columns[i] + '_' + str(v)
                attrs.append(name)
                attrs_map[name] = k
                k += 1

    attrs.append('label')
    ret_train, ret_test = [attrs], [attrs]
    for d in data_train:
        row = [0] * k
        for i in range(n - 1):
            if i in num_idx:
                if i not in attr_splits:
                    continue
                values = []
                for j in attr_splits[i]:
                    try:
                        values.append(float(j))
                    except:
                        pass
                values.sort()
                for j in range(len(values)):
                    if j == 0:
                        l, r = float('-inf'), values[j]
                    else:
                        l, r = values[j - 1], values[j]
                    try:
                        if l < d[i] <= r:
                            name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                            row[attrs_map[name]] = 1
                    except:
                        name = columns[i] + '_' + d[i]
                        row[attrs_map[name]] = 1
                l, r = values[-1] if len(values) > 0 else float('-inf'), float('inf')
                try:
                    if l < d[i] <= r:
                        name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                        row[attrs_map[name]] = 1
                except:
                    pass
            else:
                name = columns[i] + '_' + d[i]
                row[attrs_map[name]] = 1
        row.append(d[-1])
        ret_train.append(row)

    for d in data_test:
        row = [0] * k
        for i in range(n - 1):
            if i in num_idx:
                if i not in attr_splits:
                    continue
                values = []
                for j in attr_splits[i]:
                    try:
                        values.append(float(j))
                    except:
                        pass
                values.sort()
                for j in range(len(values)):
                    if j == 0:
                        l, r = float('-inf'), values[j]
                    else:
                        l, r = values[j - 1], values[j]
                    try:
                        if l < d[i] <= r:
                            name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                            row[attrs_map[name]] = 1
                    except:
                        name = columns[i] + '_' + d[i]
                        row[attrs_map[name]] = 1
                l, r = values[-1] if len(values) > 0 else float('-inf'), float('inf')
                try:
                    if l < d[i] <= r:
                        name = columns[i] + '_' + str(round(l, 3)) + '-' + str(round(r, 3))
                        row[attrs_map[name]] = 1
                except:
                    pass
            else:
                name = columns[i] + '_' + d[i]
                row[attrs_map[name]] = 1
        row.append(d[-1])
        ret_test.append(row)

    return ret_train, ret_test


def main():
    # columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jundice','autism','used_app_before','relation']
    # data, num_idx = load_data('data/autism/autism.csv', attrs=columns, label=['label'], numerics=['age'], pos='YES')
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    data, num_idx = load_data('data/cars/cars.csv', attrs=columns, label=['label'], numerics=[], pos='positive')

    _, n = np.shape(data)
    res = encode_data(data, num_idx, columns)
    # f = open('data/autism/file2.csv', 'w')
    f = open('data/cars/file2.csv', 'w')
    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()


if __name__ == '__main__':
    main()
