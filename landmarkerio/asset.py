import abc
import os
from pathlib import Path

from flask import safe_join
from landmarkerio import CacheFile
from landmarkerio.cache import _cache_image_for_id
from landmarkerio.fit import base64_to_image


class ImageAdapter(object):
    r"""
    Abstract definition of an adapter that serves image assets
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def image_info(self, asset_id):
        pass

    @abc.abstractmethod
    def texture_file(self, asset_id):
        pass

    @abc.abstractmethod
    def thumbnail_file(self, asset_id):
        pass

    @abc.abstractmethod
    def asset_ids(self):
        pass


class MeshAdapter(object):

    @abc.abstractmethod
    def asset_ids(self):
        pass

    @abc.abstractmethod
    def mesh(self, asset_id):
        pass


class CacheAdapter(object):

    def __init__(self, cache_dir):
        self.cache_dir = Path(os.path.abspath(os.path.expanduser(cache_dir)))


class ImageCacheAdapter(CacheAdapter, ImageAdapter):

    def __init__(self, cache_dir):
        CacheAdapter.__init__(self, cache_dir)
        self.regenerate_asset_ids()

    def regenerate_asset_ids(self):
        self._image_asset_ids = [a.parent.name
                                 for a in self.cache_dir.glob(os.path.join('*',
                                                              CacheFile.image))
                                 if a.parent.parent == self.cache_dir]

    def image_info(self, asset_id):
        return reduce(safe_join,
                      (str(self.cache_dir), asset_id, CacheFile.image))

    def texture_file(self, asset_id):
        return reduce(safe_join, (str(self.cache_dir),
                                  asset_id, CacheFile.texture))

    def cache_image(self, asset_id, encoded_img):
        img = base64_to_image(encoded_img)
        i = 0
        dir_ready = False
        orig_asset_id = asset_id
        while not dir_ready:
            image_dir = self.cache_dir / asset_id
            if not image_dir.is_dir():
                image_dir.mkdir()
                dir_ready = True
            else:
                asset_id = orig_asset_id + '_{}'.format(i)
                i += 1
        _cache_image_for_id(str(self.cache_dir), asset_id, img,
                            avoid_copy=False)
        self.regenerate_asset_ids()

    def thumbnail_file(self, asset_id):
        return reduce(safe_join,
                      (str(self.cache_dir), asset_id, CacheFile.thumbnail))

    def asset_ids(self):
        return self._image_asset_ids


class MeshCacheAdapter(CacheAdapter, MeshAdapter):

    def __init__(self, cache_dir):
        CacheAdapter.__init__(self, cache_dir)
        self._mesh_asset_ids = [a.parent.name
                                for a in self.cache_dir.glob(os.path.join('*',
                                                             CacheFile.mesh))
                                if a.parent.parent == self.cache_dir]

    def mesh(self, asset_id):
        return reduce(safe_join, (str(self.cache_dir), asset_id,
                                  CacheFile.mesh))

    def asset_ids(self):
        return self._mesh_asset_ids
