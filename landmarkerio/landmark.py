import abc
import json
import os.path as p
from pathlib import Path
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

    def load_lm(self, asset_id, lm_id):
        fp = self.lm_fp(asset_id, lm_id)
        if not p.isfile(fp):
            raise IOError
        with open(fp, 'rb') as f:
            lm = json.load(f)
            return lm

    def save_lm(self, asset_id, lm_id, lm_json):
        r"""
        Persist a given landmark definition to disk.
        """
        fp = self.lm_fp(asset_id, lm_id)
        with open(fp, 'wb') as f:
            json.dump(lm_json, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    @abc.abstractmethod
    def lm_fp(self, asset_id, lm_id):
        # where a landmark should exist
        pass


class SeparateDirFileLmAdapter(FileLmAdapter):

    def __init__(self, lm_dir):
        if lm_dir is None:
            # By default place the landmarks in the cwd
            lm_dir = p.join(os.getcwd(), LM_DIRNAME)
        self.lm_dir = p.abspath(p.expanduser(lm_dir))
        if not p.isdir(self.lm_dir):
            print("Warning the landmark dir does not exist - creating...")
            os.mkdir(self.lm_dir)
        print('landmarks: {}'.format(self.lm_dir))

    def lm_fp(self, asset_id, lm_id):
        # where a landmark should exist
        return safe_join(safe_join(self.lm_dir, asset_id), lm_id + FileExt.lm)

    def lm_ids(self, asset_id):
        r"""
        Return
        """
        lm_files = self._lm_paths(asset_id=asset_id)
        return [p.splitext(p.split(f)[-1])[0] for f in lm_files]

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

    def _lm_paths(self, asset_id=None):
        # what landmarks do exist and where
        if asset_id is None:
            asset_id = '*'
        g = glob.glob(p.join(safe_join(self.lm_dir, asset_id), '*'))
        return filter(lambda f: p.isfile(f) and
                                p.splitext(f)[-1] == FileExt.lm, g)

    def save_lm(self, asset_id, lm_id, lm_json):
        r"""
        Persist a given landmark definition to disk.
        """
        subject_dir = safe_join(self.lm_dir, asset_id)
        if not p.isdir(subject_dir):
            os.mkdir(subject_dir)
        super(SeparateDirFileLmAdapter, self).save_lm(asset_id, lm_id, lm_json)


class InplaceFileLmAdapter(FileLmAdapter):

    def __init__(self, asset_ids_to_paths):
        self.ids_to_paths = asset_ids_to_paths
        print('landmarks served inplace - found {} asset with '
              'landmarks'.format(len(self.asset_id_to_lm_id())))

    def lm_ids(self, asset_id):
        # always the same!
        if asset_id in self.ids_to_paths:
            return ['inplace']
        else:
            raise ValueError

    def asset_id_to_lm_id(self):
        r"""
        Return a dict mapping asset ID's to landmark IDs that are
        present on this server for that asset.
        """
        return {aid: ['inplace']
                for aid in self.ids_to_paths
                if self._lm_path_for_asset_id(aid).is_file()}

    def lm_fp(self, asset_id, lm_id):
        # note the lm_id is ignored. We just always return the .ljson file.
        return str(self._lm_path_for_asset_id(asset_id))

    def _lm_path_for_asset_id(self, asset_id):
        asset_path = Path(self.ids_to_paths[asset_id])
        return asset_path.with_suffix(FileExt.lm)
