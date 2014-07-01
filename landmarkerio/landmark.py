import abc
import json
import os.path as p
import os
from collections import defaultdict
from flask import safe_join
import glob

from landmarkerio import LM_DIR


class LmAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API
    for Landmarks.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def all_lms(self):
        pass

    @abc.abstractmethod
    def lm_ids(self, asset_id):
        pass

    @abc.abstractmethod
    def load_lm(self, asset_id, lm_id):
        pass

    @abc.abstractmethod
    def save_lm(self, asset_id, lm_id, lm_json):
        pass


class FileLmAdapter(LmAdapter):

    def __init__(self, lm_dir):
        if lm_dir is None:
            # By default place the landmarks in the cwd
            lm_dir = p.join(os.getcwd(), LM_DIR)
        self.lm_dir = p.abspath(p.expanduser(lm_dir))
        if not p.isdir(self.lm_dir):
            print("Warning the landmark dir does not exist - creating...")
            os.mkdir(self.lm_dir)
        print ('landmarks: {}'.format(self.lm_dir))

    def all_lms(self):
        lm_files = self.lm_paths()
        mapping = defaultdict(list)
        for lm_path in lm_files:
            dir_path, filename = p.split(lm_path)
            lm_set = p.splitext(filename)[0]
            lm_id = p.split(dir_path)[1]
            mapping[lm_id].append(lm_set)
        return mapping

    def lm_ids(self, asset_id):
        lm_files = self.lm_paths(asset_id=asset_id)
        return [p.splitext(p.split(f)[-1])[0] for f in lm_files]

    def save_lm(self, asset_id, lm_id, lm_json):
        subject_dir = safe_join(self.lm_dir, asset_id)
        if not p.isdir(subject_dir):
            os.mkdir(subject_dir)
        fp = self.lm_fp(asset_id, lm_id)
        with open(fp, 'wb') as f:
            json.dump(lm_json, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    def lm_fp(self, asset_id, lm_id):
        return safe_join(safe_join(self.lm_dir, asset_id),
                         lm_id + '.json')

    def lm_paths(self, asset_id=None):
        if asset_id is None:
            asset_id = '*'
        g = glob.glob(p.join(safe_join(self.lm_dir, asset_id), '*'))
        return filter(lambda f: p.isfile(f) and
                                p.splitext(f)[-1] == '.json', g)

    def load_lm(self, asset_id, lm_id):
        fp = self.lm_fp(asset_id, lm_id)
        if not p.isfile(fp):
            raise IOError
        with open(fp, 'rb') as f:
            lm = json.load(f)
            return lm


class GitLmAdapter(FileLmAdapter):

    def __init__(self, lm_dir=None):
        FileLmAdapter.__init__(self, lm_dir=lm_dir)
