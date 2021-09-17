'''
This file encodes tabular data using one hot encoding with interval merging:
continuous values of a numeric feature with same label would be merged as an interval.
'''
import numpy as np


def load_data(file, attrs=[]):
    f = open(file, 'r')
    indexes = []
    ret, i = [], 0
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            indexes = [j for j in range(len(line)) if line[j] in attrs]
        else:
            line = line.strip('\n').split(',')
            r = [line[j] for j in indexes]
            ret.append(r)
        i += 1
    return ret


def discretize(nums, y):
    values_dict = dict()
    str_set = set()
    values = []
    for i in range(len(nums)):
        try:
            if values_dict.get(float(nums[i])) is None:
                values_dict[float(nums[i])] = y[i][0]
            elif values_dict[float(nums[i])] != y[i][0]:
                values_dict[float(nums[i])] = -1
            values.append(float(nums[i]))
        except:
            str_set.add(nums[i])
    values = list(set(values))
    values.sort()

    val_list, ret = [], []
    for i in range(len(values)):
        if len(val_list) == 0 or values_dict[values[i]] < 0:
            val_list.append(values[i])
            continue
        last_val = val_list[-1]
        if values_dict[last_val] < 0 or values_dict[values[i]] != values_dict[last_val]:
            val_list.append(values[i])

    ret.append((float('-inf'), val_list[0]))
    for i in range(len(val_list) - 1):
        ret.append((val_list[i], val_list[i + 1]))
    ret.append((val_list[-1], float('inf')))
    return ret, list(str_set)


def encode_data(y, data, attrs=[], numerics=[]):
    r, c = np.shape(data)
    mapped_attrs = dict()
    mapped_numerics = dict()
    n = 0
    for i in range(c):
        if attrs[i] in numerics:
            nums = [data[j][i] for j in range(r)]
            pairs, strs = discretize(nums, y)
            mapped_numerics[i] = pairs
            for p in pairs:
                name = attrs[i] + '_' + str(round(p[0], 3)) + '-' + str(round(p[1], 3))
                if name not in mapped_attrs:
                    mapped_attrs[name] = n
                    n += 1
            for s in strs:
                name = attrs[i] + '_' + s
                if name not in mapped_attrs:
                    mapped_attrs[name] = n
                    n += 1
        else:
            values = set()
            for j in range(r):
                values.add(data[j][i])
            for v in values:
                name = attrs[i] + '_' + v
                if name not in mapped_attrs:
                    mapped_attrs[name] = n
                    n += 1
    ret = []
    for i in range(r):
        line = [0] * len(mapped_attrs)
        for j in range(c):
            if attrs[j] not in numerics:
                name = attrs[j] + '_' + data[i][j]
                line[mapped_attrs[name]] = 1
            else:
                rgs = mapped_numerics[j]
                for p in rgs:
                    try:
                        if p[0] <= float(data[i][j]) <= p[1]:
                            name = attrs[j] + '_' + str(round(p[0], 3)) + '-' + str(round(p[1], 3))
                            line[mapped_attrs[name]] = 1
                    except:
                        name = attrs[j] + '_' + data[i][j]
                        line[mapped_attrs[name]] = 1
        ret.append(line)
    return mapped_attrs, ret


def encode_label(data, pos):
    r, c = np.shape(data)
    ret = []
    for i in range(r):
        line = []
        for j in range(c):
            if data[i][j] == pos:
                line.append(1)
            else:
                line.append(0)
        ret.append(line)
    return ret


def convert_data(file, columns=[], label='', pos = '', numerics=[]):
    data = load_data(file, [label])
    y = encode_label(data, pos)
    data = load_data(file, columns)
    mapped_attrs, x = encode_data(y, data, columns, numerics)
    mapped_attrs[label] = len(mapped_attrs)
    data = np.append(x, y, axis=1)
    attrs = [''] * len(mapped_attrs)
    for k in mapped_attrs:
        attrs[mapped_attrs[k]] = k
    return attrs, data


def main():
    # columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','age','gender','ethnicity','jaundice','autism','used_app_before','relation']
    # attrs, data = convert_data('data/autism/autism.csv', columns, 'label', 'YES', numerics=['age'])
    columns = ['buying', 'maint', 'doors', 'persons', 'lugboot', 'safety']
    attrs, data = convert_data('data/cars/cars.csv', columns, 'label', 'positive', numerics=['a1'])
    res = [attrs]
    for d in data:
        res.append(d)
    # f = open('data/autism/data_train.csv', 'w')
    f = open('data/cars/file.csv', 'w')
    for item in res:
        f.write(','.join([str(x) for x in item]) + '\n')
    f.close()


if __name__ == '__main__':
    main()
