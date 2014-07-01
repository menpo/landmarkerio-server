import itertools
from collections import namedtuple
import os.path as p
import glob
from flask import safe_join
import abc

from landmarkerio import TEMPLATE_DINAME, TEMPLATE_EXT


Group = namedtuple('Group', ['label', 'n', 'index'])


def parse_group(group):
    # split on \n and strip left and right whitespace.
    x = [l.strip() for l in group.split('\n')]
    label, n_str = x[0].split(' ')
    n = int(n_str)
    index_str = x[1:]
    if len(index_str) == 0:
        return Group(label, n, [])
    index = []
    for i in index_str:
        if ':' in i:
            # User is providing a slice
            start, end = (int(x) for x in i.split(':'))
            index.extend([x, x+1] for x in xrange(start, end))
        else:
            # Just a standard pair of numbers
            index.append([int(j) for j in i.split(' ')])
    indexes = set(itertools.chain.from_iterable(index))
    if min(indexes) < 0 or max(indexes) > n:
        raise ValueError("invalid connectivity")
    return Group(label, n, index)


def group_to_json(group, n_dims):
    group_json = {}
    lms = [{'point': [None] * n_dims}] * group.n
    group_json['landmarks'] = lms
    group_json['connectivity'] = group.index
    group_json['label'] = group.label
    return group_json


def groups_to_json(groups, n_dims):
    lm_json = {'version': 1, 'groups': []}
    for g in groups:
        lm_json['groups'].append(group_to_json(g, n_dims))
    return lm_json


def load_template(path, n_dims):
    with open(path, 'rb') as f:
        ta = f.read().strip().split('\n\n')
    groups = [parse_group(g) for g in ta]
    return groups_to_json(groups, n_dims)


class TemplateAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API
    for Template retrieval.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def templates(self):
        pass

    @abc.abstractmethod
    def template_json(self, lm_id):
        pass


class FileTemplateAdapter(TemplateAdapter):

    def __init__(self, n_dims, template_dir=None):
        self.n_dims = n_dims
        if template_dir is None:
            # try the user folder
            user_templates = p.expanduser(p.join('~', TEMPLATE_DINAME))
            if p.isdir(user_templates):
                template_dir = user_templates
            else:
                raise ValueError("No template dir provided and "
                                 "{} doesn't exist".format(user_templates))
        self.template_dir = p.abspath(p.expanduser(template_dir))
        print ('templates: {}'.format(self.template_dir))

    def templates(self):
        template_paths = glob.glob(p.join(self.template_dir,
                                          '*' + TEMPLATE_EXT))
        return [p.splitext(p.split(t)[-1])[0] for t in template_paths]

    def template_json(self, lm_id):
        fp = safe_join(self.template_dir, lm_id + TEMPLATE_EXT)
        return load_template(fp, self.n_dims)
