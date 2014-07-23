import abc
import os
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


class MeshAdapter(ImageAdapter):

    @abc.abstractmethod
    def mesh_json(self, asset_id):
        pass


class CacheAdapter(object):

    def __init__(self, cache_dir):
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))


class ImageCacheAdapter(CacheAdapter, ImageAdapter):

    def image_info(self, asset_id):
        return reduce(safe_join,
                      (self.cache_dir, asset_id, IMAGE_INFO_FILENAME))

    def texture_file(self, asset_id):
        return reduce(safe_join, (self.cache_dir, asset_id, TEXTURE_FILENAME))

    def thumbnail_file(self, asset_id):
        return reduce(safe_join,
                      (self.cache_dir, asset_id, THUMBNAIL_FILENAME))

    def asset_ids(self):
        return os.listdir(self.cache_dir)


class MeshCacheAdapter(ImageCacheAdapter, MeshAdapter):

    def mesh_json(self, asset_id):
        return reduce(safe_join, (self.cache_dir, asset_id, MESH_FILENAME))
