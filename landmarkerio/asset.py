import abc
import os
from pathlib import Path

from flask import safe_join
from landmarkerio import (IMAGE_INFO_FILENAME, TEXTURE_FILENAME,
                          THUMBNAIL_FILENAME, MESH_FILENAME)


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
    def mesh_json(self, asset_id):
        pass


class CacheAdapter(object):

    def __init__(self, cache_dir):
        self.cache_dir = Path(os.path.abspath(os.path.expanduser(cache_dir)))


class ImageCacheAdapter(CacheAdapter, ImageAdapter):

    def __init__(self, cache_dir):
        CacheAdapter.__init__(self, cache_dir)
        self._image_asset_ids = [a.parent.name
                                 for a in self.cache_dir.glob("*/image.json")
                                 if a.parent.parent == self.cache_dir]

    def image_info(self, asset_id):
        return reduce(safe_join,
                      (str(self.cache_dir), asset_id, IMAGE_INFO_FILENAME))

    def texture_file(self, asset_id):
        return reduce(safe_join, (str(self.cache_dir),
                                  asset_id, TEXTURE_FILENAME))

    def thumbnail_file(self, asset_id):
        return reduce(safe_join,
                      (str(self.cache_dir), asset_id, THUMBNAIL_FILENAME))

    def asset_ids(self):
        return self._image_asset_ids


class MeshCacheAdapter(CacheAdapter, MeshAdapter):

    def __init__(self, cache_dir):
        CacheAdapter.__init__(self, cache_dir)
        self._mesh_asset_ids = [a.parent.name
                                for a in self.cache_dir.glob("*/mesh.json.gz")
                                if a.parent.parent == self.cache_dir]

    def mesh_json(self, asset_id):
        return reduce(safe_join, (str(self.cache_dir), asset_id,
                                  MESH_FILENAME))

    def asset_ids(self):
        return self._mesh_asset_ids
