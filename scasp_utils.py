import numpy as np
import random
import subprocess
import tempfile


def load_data(file, numerics, amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx = [], []
    ret, i, k = [], 0, 0
    attrs = []
    for line in f.readlines():
        line = line.strip('\n').split(',')
        if i == 0:
            attrs = [line[j].lower().replace(' ', '_') for j in range(len(line))]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
        else:
            r = [j for j in range(len(line))]
            for j in range(len(line)):
                if j in num_idx:
                    try:
                        r[j] = float(line[j])
                    except:
                        r[j] = line[j]
                else:
                    r[j] = line[j]
            ret.append(r)
        i += 1
        amount -= 1
        if amount == 0:
            break
    return ret, attrs


def decode_data(data, attrs, seq=0, label_flag=False):
    ret = []
    _, n = np.shape(data)
    i = seq
    if label_flag:
        n += 1
    for d in data:
        line = ['id(' + str(i) + ').']
        for j in range(n - 1):
            if isinstance(d[j], float) or isinstance(d[j], int):
                pred = attrs[j].lower() + '(' + str(i) + ',' + str(d[j]) + ').'
            elif len(d[j]) > 0:
                pred = attrs[j].lower() + '(' + str(i) + ',\'' + \
                       str(d[j]).lower().replace(' ', '_').replace('\'', '')\
                           .replace('\"', '').replace('.', '') + '\').'
            else:
                pred = attrs[j].lower() + '(' + str(i) + ',\'' + 'null' + '\').'
            line.append(pred)
        ret.append(line)
        i += 1
    return ret


def load_data_pred(file, numerics, seq=0, label_flag=False, amount=-1):
    data, attrs = load_data(file, numerics=numerics, amount=amount)
    data_pred = decode_data(data, attrs, seq=seq,label_flag=label_flag)
    return data_pred


def split_data_pred(X, Y, X_pred, ratio=0.8, rand=True):
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
    X_test_pred = [X_pred[i] for i in range(n) if i not in set(train)]
    return X_train, Y_train, X_test_pred


def load_translation(model, file):
    model.translation = []
    f = open(file, 'r')
    for line in f.readlines():
        model.translation.append(line.strip('\n'))


def save_asp_to_file(model, file):
    if model.asp() is None:
        return
    f = open(file, 'w')
    for r in model.asp_rules:
        f.write(r + '\n')
    f.close()


def scasp_query(model, x, pred=True):
    if model.asp() is None:
        return
    tf = tempfile.NamedTemporaryFile()
    for r in model.asp_rules:
        tf.write((r + '\n').encode())
    if model.translation is not None:
        for t in model.translation:
            tf.write((t + '\n').encode())
    if not pred:
        data_pred = decode_data([x], model.attrs, model.seq)
        for preds in data_pred:
            for p in preds:
                tf.write((p + '\n').encode())
        seq = str(model.seq)
        model.seq += 1
    else:
        for d in x:
            tf.write((d + '\n').encode())
        first = x[0].replace('(', '#').replace(',', '#').replace(')', '#').split('#')
        seq = first[1]

    extra = 'goal(X):-' + model.label.lower().replace(' ', '_') + '(X,\'' + model.pos.lower().replace(' ', '_') + '\').\n'
    tf.write(extra.encode())
    query = '?- goal(' + seq + ').'
    tf.write(query.encode())
    tf.flush()

    command = 'scasp' + ' -s1 --tree --human --pos ' + tf.name
    res = subprocess.run([command], stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    tf.close()
    return res
