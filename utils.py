import itertools
from collections import namedtuple

Group = namedtuple('Group', ['label', 'n', 'index'])


def parse_group(group):
    x = group.split('\n')
    label, n_str = x[0].split(' ')
    n = int(n_str)
    index_str = x[1:]
    if len(index_str) == 0:
        return Group(label, n, [])
    index = [[int(j) for j in i.split(' ')] for i in index_str]
    indexes = set(itertools.chain.from_iterable(index))
    if min(indexes) < 0 or max(indexes) > n:
        raise ValueError("invalid connectivity")
    return Group(label, n, index)


def group_to_json(group):
    group_json = {}
    lms = [{'point': None}] * group.n
    group_json["landmarks"] = lms
    group_json["connectivity"] = group.index
    return group_json


def groups_to_json(groups):
    lm_json = {'version': 1, 'groups': {}}
    for g in groups:
        lm_json['groups'][g.label] = group_to_json(g)
    return lm_json


def load_template(path):
    with open(path, 'rb') as f:
        ta = f.read().strip().split('\n\n')
    print ta
    groups = [parse_group(g) for g in ta]
    return groups_to_json(groups)
