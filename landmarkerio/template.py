import itertools
from collections import namedtuple
import os.path as p
from pathlib import Path
from flask import safe_join
import abc
import yaml

from landmarkerio import TEMPLATE_DINAME, FileExt


Group = namedtuple('Group', ['label', 'n', 'index'])


def parse_connectivity(index_lst, n):
    index = []
    for i in index_lst:
        if ':' in i:
            # User is providing a slice
            start, end = (int(x) for x in i.split(':'))
            index.extend([x, x+1] for x in xrange(start, end))
        else:
            # Just a standard pair of numbers
            index.append([int(j) for j in i.split(' ')])

    indexes = set(itertools.chain.from_iterable(index))
    if len(index) > 0 and (min(indexes) < 0 or max(indexes) > n):
        raise ValueError("invalid connectivity")

    return index


def load_yaml_template(filepath, n_dims):
    with open(filepath) as f:
        data = yaml.load(f.read())

        if 'groups' in data:
            raw_groups = data['groups']
        else:
            raise KeyError(
                "Missing 'groups' or 'template' key in yaml file %s"
                % filepath)

        groups = []

        for index, group in enumerate(raw_groups):

            label = group.get('label', index)  # Allow simple ordered groups

            n = group['points']  # Should raise KeyError by design if missing
            connectivity = group.get('connectivity', [])

            if isinstance(connectivity, list):
                index = parse_connectivity(connectivity, n)
            elif connectivity is 'cycle':
                index = parse_connectivity(
                    ['0:%d' % (n - 1), '%d 0' % (n - 1)], n)
            else:
                index = []  # Couldn't parse connectivity, safe default

            groups.append(Group(label, n, index))

    return build_json(groups, n_dims)


def parse_group(group):
    # split on \n and strip left and right whitespace.
    x = [l.strip() for l in group.split('\n')]
    label, n_str = x[0].split(' ')
    n = int(n_str)
    index_str = x[1:]
    if len(index_str) == 0:
        return Group(label, n, [])
    index = parse_connectivity(index_str, n)
    return Group(label, n, index)


def group_to_json(group, n_dims):
    group_json = {}
    lms = [{'point': [None] * n_dims}] * group.n
    group_json['landmarks'] = lms
    group_json['connectivity'] = group.index
    group_json['label'] = group.label
    return group_json


def build_json(groups, n_dims):
    n_points = sum(g.n for g in groups)
    offset = 0
    connectivity = []
    labels = []
    for g in groups:
        connectivity += [[j + offset for j in i] for i in g.index]
        labels.append({
            'label': g.label,
            'mask': list(range(offset, offset + g.n))
        })
        offset += g.n

    lm_json = {
        'labels': labels,
        'landmarks': {
            'connectivity': connectivity,
            'points': [[None] * n_dims] * n_points
        },
        'version': 2,
    }

    return lm_json


def load_legacy_template(path, n_dims):
    with open(path) as f:
        ta = f.read().strip().split('\n\n')
    groups = [parse_group(g) for g in ta]
    return build_json(groups, n_dims)


def load_template(path, n_dims):
    if path.endswith('yaml') or path.endswith('yml'):
        return load_yaml_template(path, n_dims)
    else:
        return load_legacy_template(path, n_dims)


class TemplateAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API
    for Template retrieval.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def template_ids(self):
        pass

    @abc.abstractmethod
    def load_template(self, lm_id):
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
        self.template_dir = Path(p.abspath(p.expanduser(template_dir)))
        print ('templates: {}'.format(self.template_dir))

    def template_ids(self):
        return [t.stem for t in self.template_paths()]

    def template_paths(self):
        return self.template_dir.glob('*' + FileExt.template)

    def load_template(self, lm_id):
        fp = safe_join(str(self.template_dir), lm_id + FileExt.template)
        return load_template(fp, self.n_dims)


class CachedFileTemplateAdapter(FileTemplateAdapter):

    def __init__(self, n_dims, template_dir=None):
        super(CachedFileTemplateAdapter,
              self).__init__(n_dims, template_dir=template_dir)
        self._cache = {lm_id: FileTemplateAdapter.load_template(self, lm_id)
                       for lm_id in FileTemplateAdapter.template_ids(self)}
        print('cached {} templates ({})'.format(
            len(self._cache), ', '.join(self._cache.keys())))

    def load_template(self, lm_id):
        return self._cache[lm_id]
