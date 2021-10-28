import numpy as np
import pickle


def load_data(file, attrs, label, numerics, pos):
    f = open(file, 'r')
    attr_idx, num_idx, label_idx = [], [], -1
    ret, i = [], 0
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            attr_idx = [j for j in range(len(line)) if line[j] in attrs]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
            for j in range(len(line)):
                if line[j] == label:
                    label_idx = j
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
            if label_idx != -1:
                y = 1 if line[label_idx] == pos else 0
                r.append(y)
            ret.append(r)
        i += 1
    label = label + '#==#' + pos
    attrs.append(label)
    return ret, attrs


def evaluate(d, lit):
    i, r, v = lit[0], lit[1], lit[2]
    if isinstance(v, str):
        if r == '==':
            return d[i] == v
        elif r == '!=':
            return d[i] != v
        else:
            return False
    elif isinstance(d[i], str):
        return False
    elif r == '<=':
        return d[i] <= v
    elif r == '>':
        return d[i] > v
    else:
        return False


def build_table(data):
    _, n = np.shape(data)
    table = dict()
    for i in range(n - 1):
        table[i] = set()
        for d in data:
            table[i].add(d[i])
    return table


def onehot_encode(data, attrs, table=None, attrs_map=None, label_flag=True):
    m, n = np.shape(data)
    if table is None:
        table = build_table(data)
    if attrs_map is None:
        attrs_map = dict()
        k = 0
        for i in range(n - 1):
            if i not in table:
                continue
            xs, cs = [], []
            for j in table[i]:
                if isinstance(j, str):
                    cs.append(j)
                else:
                    xs.append(j)
                xs.sort()
            for j in cs:
                attrs_map[(i, '==', j)] = k
                k += 1
            if len(xs) > 0:
                xs = [float('-inf')] + xs + [float('inf')]
                for j in range(1, len(xs)):
                    attrs_map[((i, '>', xs[j - 1]), (i, '<=', xs[j]))] = k
                    k += 1
    w = len(attrs_map)
    ret = []
    first_line = [0] * w
    for l in attrs_map:
        if len(l) == 3:
            i, r, v = l[0], l[1], l[2]
            j = attrs_map[l]
            key = attrs[i] + '#' + str(r) + '#' + str(v)
            first_line[j] = key
        elif len(l) == 2:
            j = attrs_map[l]
            left, right = l[0], l[1]
            lval, rval = left[2], right[2]
            i = left[0]
            key = str(lval) + '#' + '<' + '#' + attrs[i] + '#' + '<=' + '#' + str(rval)
            first_line[j] = key
    if label_flag:
        first_line.append(attrs[-1])
    first_line = [s.lower().replace(' ', '_') for s in first_line]
    ret.append(first_line)
    for d in data:
        line = [0] * w
        for l in attrs_map:
            if len(l) == 3:
                j = attrs_map[l]
                line[j] = 1 if evaluate(d, l) else 0
            elif len(l) == 2:
                j = attrs_map[l]
                left, right = l[0], l[1]
                line[j] = 1 if evaluate(d, left) and evaluate(d, right) else 0
        if label_flag:
            line.append(d[-1])
        ret.append(line)
    return ret, attrs_map


class OneHotEncoder(object):
    def __init__(self, attrs, numerics, label, pos):
        self.attrs = attrs
        self.numerics = numerics
        self.label = label
        self.pos = pos
        self.table = None
        self.attrs_map = None

    def encode(self, file, label_flag=True):
        data, self.attrs = load_data(file, attrs=self.attrs, label=self.label, numerics=self.numerics, pos=self.pos)
        if self.table is None:
            self.table = build_table(data)
            mat, self.attrs_map = onehot_encode(data, self.attrs, table=self.table, label_flag=label_flag)
        else:
            mat, _ = onehot_encode(data, self.attrs, table=self.table, attrs_map=self.attrs_map, label_flag=label_flag)
        return mat


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


def entropy(data, i=-1):
    ret = 0.0
    if len(data) == 0:
        return ret
    n, _ = np.shape(data)
    xs = dict()
    for d in data:
        if xs.get(d[i]) is None:
            xs[d[i]] = 0.0
        xs[d[i]] += 1.0
    for x in xs:
        xs[x] /= n
        if xs[x]:
            ret -= xs[x] * np.log(xs[x])
    return ret


def best_ig(data, i):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()
    for d in data:
        if pos.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            if d[-1]:
                pos[d[i]] += 1.0
                cp += 1.0
            else:
                neg[d[i]] += 1.0
                cn += 1.0
        else:
            xs.add(d[i])
            if d[-1]:
                pos[d[i]] += 1.0
                xp += 1.0
            else:
                neg[d[i]] += 1.0
                xn += 1.0
    xs = list(xs)
    xs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]
    best, v = float('-inf'), float('-inf')
    for x in xs:
        ifg = ig(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x])
        if best < ifg:
            best, v = ifg, x
    for c in cs:
        ifg = ig(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
        if best < ifg:
            best, v = ifg, c
    return entropy(data) + best, v


class Node(object):
    def __init__(self, data):
        self.data = data
        self.index = -1
        self.pivot = -1
        self.children = dict()


def all_same(data, i=-1):
    xs = set()
    for d in data:
        xs.add(d[i])
    return len(xs) <= 1


def split_node(node):
    data = node.data
    if len(data) == 0 or entropy(data) == 0:
        return
    _, n = np.shape(data)
    max_ig = float('-inf')
    best_idx, best_v = -1, ''
    for i in range(0, n - 1):
        ifg, v = best_ig(data, i)
        if max_ig < ifg:
            max_ig = ifg
            best_v, best_idx = v, i
    if max_ig == 0:
        return
    node.index = best_idx
    node.pivot = best_v
    left_data, right_data = [], []
    if isinstance(best_v, str):
        left_name, right_name = '==', '!='
        for d in data:
            if evaluate(d, (best_idx, '==', best_v)):
                left_data.append(d)
            else:
                right_data.append(d)
    else:
        left_name, right_name = '<=', '>'
        for d in data:
            if evaluate(d, (best_idx, '<=', best_v)):
                left_data.append(d)
            else:
                right_data.append(d)
    if len(left_data) == 0 or len(right_data) == 0:
        return
    node.children[left_name] = Node(left_data)
    node.children[right_name] = Node(right_data)
    for i in node.children:
        split_node(node.children[i])


def build_tree(data):
    ret = Node(data)
    split_node(ret)
    return ret


def traverse(node, path, splits):
    if node is None:
        return
    if all_same(node.data):
        return
    for i in node.children:
        path.append((node.index, node.pivot, i))
        if splits.get(node.index) is None:
            splits[node.index] = set()
        splits[node.index].add(node.pivot)
        traverse(node.children[i], path, splits)
        path.pop()


def tree_encode(data, attrs, root=None, attrs_map=None, label_flag=True):
    m, n = np.shape(data)
    splits = dict()
    if root is None:
        root = build_tree(data)
    traverse(root, [], splits)

    if attrs_map is None:
        for d in data:
            for i in range(n - 1):
                if i not in splits:
                    continue
                if isinstance(d[i], str):
                    splits[i].add(d[i])
        attrs_map = dict()
        k = 0
        for i in range(n - 1):
            if i not in splits:
                continue
            xs, cs = [], []
            for j in splits[i]:
                if isinstance(j, str):
                    cs.append(j)
                else:
                    xs.append(j)
                xs.sort()
            for j in cs:
                attrs_map[(i, '==', j)] = k
                k += 1
            if len(xs) > 0:
                xs = [float('-inf')] + xs + [float('inf')]
                for j in range(1, len(xs)):
                    attrs_map[((i, '>', xs[j - 1]), (i, '<=', xs[j]))] = k
                    k += 1

    w = len(attrs_map)
    ret = []
    first_line = [0] * w
    for l in attrs_map:
        if len(l) == 3:
            i, r, v = l[0], l[1], l[2]
            j = attrs_map[l]
            key = attrs[i] + '#' + str(r) + '#' + str(v)
            first_line[j] = key
        elif len(l) == 2:
            j = attrs_map[l]
            left, right = l[0], l[1]
            lval, rval = left[2], right[2]
            i = left[0]
            key = str(lval) + '#' + '<' + '#' + attrs[i] + '#' + '<=' + '#' + str(rval)
            first_line[j] = key
    if label_flag:
        first_line.append(attrs[-1])
    first_line = [s.lower().replace(' ', '_') for s in first_line]
    ret.append(first_line)
    for d in data:
        line = [0] * w
        for l in attrs_map:
            if len(l) == 3:
                j = attrs_map[l]
                line[j] = 1 if evaluate(d, l) else 0
            elif len(l) == 2:
                j = attrs_map[l]
                left, right = l[0], l[1]
                line[j] = 1 if evaluate(d, left) and evaluate(d, right) else 0
        if label_flag:
            line.append(d[-1])
        ret.append(line)
    return ret, attrs_map


class TreeEncoder(object):
    def __init__(self, attrs, numerics, label, pos):
        self.attrs = attrs
        self.numerics = numerics
        self.label = label
        self.pos = pos
        self.root = None
        self.attrs_map = None

    def encode(self, file, label_flag=True):
        data, self.attrs = load_data(file, attrs=self.attrs, label=self.label, numerics=self.numerics, pos=self.pos)
        if self.root is None:
            self.root = build_tree(data)
            mat, self.attrs_map = tree_encode(data, self.attrs, root=self.root, label_flag=label_flag)
        else:
            mat, _ = tree_encode(data, self.attrs, root=self.root, attrs_map=self.attrs_map, label_flag=label_flag)
        return mat


def save_data_to_file(data, file):
    f = open(file, 'w')
    for d in data:
        f.write(','.join([str(x) for x in d]) + '\n')
    f.close()


def save_model_to_file(model, file):
    f = open(file, 'wb')
    pickle.dump(model, f)
    f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret


def main():
    attrs = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'age', 'gender', 'ethnicity', 'jaundice',
             'pdd', 'used_app_before', 'relation']
    nums = ['age']

    encoder = OneHotEncoder(attrs=attrs, numerics=nums, label='label', pos='NO')
    mat = encoder.encode('data/autism/autism.csv')
    save_data_to_file(mat, 'file.csv')

    encoder = TreeEncoder(attrs=attrs, numerics=nums, label='label', pos='NO')
    mat = encoder.encode('data/autism/autism.csv')
    save_data_to_file(mat, 'file2.csv')


if __name__ == '__main__':
    main()
