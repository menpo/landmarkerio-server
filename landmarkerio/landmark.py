import abc
import json
import os.path as p
import os
from collections import defaultdict
from flask import safe_join
import glob

from landmarkerio import LM_DIRNAME, FileExt


class LmAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API
    for Landmarks.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def asset_id_to_lm_id(self):
        pass

    @abc.abstractmethod
    def lm_ids(self, asset_id):
        r"""
        Return a list of lm_ids available for a given asset_id
        """
        pass

    @abc.abstractmethod
    def load_lm(self, asset_id, lm_id):
        pass

    @abc.abstractmethod
    def save_lm(self, asset_id, lm_id, lm_json):
        pass


class FileLmAdapter(LmAdapter):
    r"""
    Concrete implementation of LmAdapater that serves landmarks from the
    local filesystem.
    """

    def __init__(self, lm_dir):
        if lm_dir is None:
            # By default place the landmarks in the cwd
            lm_dir = p.join(os.getcwd(), LM_DIRNAME)
        self.lm_dir = p.abspath(p.expanduser(lm_dir))
        if not p.isdir(self.lm_dir):
            print("Warning the landmark dir does not exist - creating...")
            os.mkdir(self.lm_dir)
        print ('landmarks: {}'.format(self.lm_dir))

    def asset_id_to_lm_id(self):
        r"""
        Return a dict mapping asset ID's to landmark IDs that are
        present on this server for that asset.
        """
        lm_files = self._lm_paths()
        mapping = defaultdict(list)
        for lm_path in lm_files:
            dir_path, filename = p.split(lm_path)
            lm_id = p.splitext(filename)[0]
            asset_id = p.split(dir_path)[1]
            mapping[asset_id].append(lm_id)
        return mapping

    def lm_ids(self, asset_id):
        r"""
        Return
        """
        lm_files = self._lm_paths(asset_id=asset_id)
        return [p.splitext(p.split(f)[-1])[0] for f in lm_files]

    def save_lm(self, asset_id, lm_id, lm_json):
        r"""
        Persist a given landmark definition to disk.
        """
        subject_dir = safe_join(self.lm_dir, asset_id)
        if not p.isdir(subject_dir):
            os.mkdir(subject_dir)
        fp = self._lm_fp(asset_id, lm_id)
        with open(fp, 'wb') as f:
            json.dump(lm_json, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    def load_lm(self, asset_id, lm_id):
        fp = self._lm_fp(asset_id, lm_id)
        if not p.isfile(fp):
            raise IOError
        with open(fp, 'rb') as f:
            lm = json.load(f)
            return lm

    def _lm_fp(self, asset_id, lm_id):
        return safe_join(safe_join(self.lm_dir, asset_id), lm_id + FileExt.lm)

    def _lm_paths(self, asset_id=None):
        if asset_id is None:
            asset_id = '*'
        g = glob.glob(p.join(safe_join(self.lm_dir, asset_id), '*'))
        return filter(lambda f: p.isfile(f) and
                                p.splitext(f)[-1] == FileExt.lm, g)
