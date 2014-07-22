import abc
import json
import os
import os.path as p
import shutil
import gzip
from flask import safe_join

import menpo.io as mio
from menpo.shape.mesh import TexturedTriMesh

from landmarkerio import (CACHE_DIRNAME, IMAGE_INFO_FILENAME, TEXTURE_FILENAME,
                          THUMBNAIL_FILENAME, MESH_FILENAME)


def asset_id_for_path(fp):
    return p.splitext(p.split(fp)[-1])[0]


def save_jpg_thumbnail_file(img, path, width=640):
    ip = img.as_PILImage()
    w, h = ip.size
    h2w = h * 1. / w
    ips = ip.resize((width, int(h2w * width)))
    ips.save(path, quality=20, format='jpeg')


class ImageAdapter(object):
    r"""
    Abstract definition of an adapter that serves image assets
    """

    @abc.abstractmethod
    def image_info(self, asset_id):
        pass

    @abc.abstractmethod
    def texture_file(self, asset_id):
        pass

    @abc.abstractmethod
    def thumbnail_file(self, asset_id):
        pass


class MeshAdapter(ImageAdapter):
    r"""
    Abstract definition of an adapter that serves mesh assets

    """

    @abc.abstractmethod
    def mesh_json(self, asset_id):
        pass


class MenpoAdapter(object):

    def __init__(self, asset_dir, recursive=False, ext=None, cache_dir=None):
        self.asset_dir = p.abspath(p.expanduser(asset_dir))
        if not p.isdir(self.asset_dir):
            raise ValueError('{} is not a directory.'.format(self.asset_dir))
        print ('assets:    {}'.format(self.asset_dir))
        self.recursive = recursive
        if self.recursive:
            print('assets dir will be searched recursively.')
        if ext is not None:
            self.extension_str = '.' + ext
            print('only assets of type {} will be '
                  'loaded.'.format(self.extension_str))
        else:
            self.extension_str = ''

        if cache_dir is None:
            # By default place the cache in the cwd
            cache_dir = p.join(os.getcwd(), CACHE_DIRNAME)
        self.cache_dir = p.abspath(p.expanduser(cache_dir))
        if not p.isdir(self.cache_dir):
            print("Warning the cache dir does not exist - creating...")
            os.mkdir(self.cache_dir)
        print ('cache:     {}'.format(self.cache_dir))

        # Construct a mapping from id's to file paths
        self.asset_paths = {}
        self._build_asset_mapping()

        # Check cache
        asset_ids = set(self.asset_paths.iterkeys())
        cached = set(os.listdir(self.cache_dir))
        uncached = asset_ids - cached
        print('{} assets need to be added to '
              'the cache'.format(len(uncached)))
        for i, asset in enumerate(uncached):
            print('Caching {}/{} - {}'.format(i + 1, len(uncached), asset))
            self.cache_asset(asset)
        if len(uncached) > 0:
                print('{} assets cached.'.format(len(uncached)))

    def cache_asset(self, asset_id):
        r"""
        Caches the info for a given asset id so it can be efficiently
        served in the future.

        Parameters
        ----------
        asset_id : `str`
        The id of the asset that needs to be cached
        """
        print('Caching asset {}'.format(asset_id))
        if not asset_id in self.asset_paths:
            raise ValueError('{} is not a valid asset_id'.format(asset_id))
        asset_cache_dir = p.join(self.cache_dir, asset_id)
        if not p.isdir(asset_cache_dir):
            print("Cache for {} does not exist - creating...".format(asset_id))
            os.mkdir(asset_cache_dir)
        self._cache_asset(asset_id)

    @property
    def _glob_pattern(self):
        file_glob = '*' + self.extension_str
        if self.recursive:
            return os.path.join('**', file_glob)
        else:
            return file_glob

    @abc.abstractmethod
    def _cache_asset(self, asset_id):
        r"""
        Actually cache an asset.

        Parameters
        ----------
        asset_id : `str`
            The id of the asset that needs to be cached
        """
        pass

    @abc.abstractmethod
    def _asset_paths(self):
        r"""
        Recalculate paths to assets that this server can import

        Returns
        -------
        paths : `iterable`
            An iterable of paths that this server can import.
        """
        return []

    def _build_asset_mapping(self):
        self.asset_paths = {}
        for path in self._asset_paths():
            asset_id = asset_id_for_path(path)
            if asset_id in self.asset_paths:
                raise RuntimeError(
                    "asset_id {} is not unique - links to {} and "
                    "{}".format(asset_id, self.asset_paths[asset_id], path))
            self.asset_paths[asset_id] = path

    def asset_ids(self):
        return self.asset_paths.keys()


class ImageMenpoAdapter(MenpoAdapter, ImageAdapter):

    def _asset_paths(self):
        return mio.image_paths(p.join(self.asset_dir, self._glob_pattern))

    def _cache_asset(self, asset_id):
        r"""Actually cache this asset_id.
        """
        img = mio.import_image(self.asset_paths[asset_id])
        self._cache_image_for_id(asset_id, img)

    def _cache_image_for_id(self, asset_id, img):
        asset_cache_dir = p.join(self.cache_dir, asset_id)
        image_info_path = p.join(asset_cache_dir, IMAGE_INFO_FILENAME)
        texture_path = p.join(asset_cache_dir, TEXTURE_FILENAME)
        thumbnail_path = p.join(asset_cache_dir, THUMBNAIL_FILENAME)
        # 1. Save out the image metadata json
        image_info = {'width': img.width,
                      'height': img.height}
        with open(image_info_path, 'wb') as f:
            json.dump(image_info, f)
        # 2. Save out the image
        if img.ioinfo.extension == '.jpg':
            # Original was a jpg, save it
            shutil.copyfile(img.ioinfo.filepath, texture_path)
        else:
            # Original wasn't a jpg - make it so
            img.as_PILImage().save(texture_path, format='jpeg')
        # 3. Save out the thumbnail
        save_jpg_thumbnail_file(img, thumbnail_path)

    def image_info(self, asset_id):
        return p.join(self.cache_dir, asset_id, IMAGE_INFO_FILENAME)

    def texture_file(self, asset_id):
        return p.join(self.cache_dir, asset_id, TEXTURE_FILENAME)

    def thumbnail_file(self, asset_id):
        return p.join(self.cache_dir, asset_id, THUMBNAIL_FILENAME)


class MeshMenpoAdapter(ImageMenpoAdapter, MeshAdapter):

    def __init__(self, asset_dir, recursive=False, ext=None, cache_dir=None):
        ImageMenpoAdapter.__init__(self, asset_dir, recursive=recursive,
                                   ext=ext, cache_dir=cache_dir)
        self.meshes = {}

    def _asset_paths(self):
        return mio.mesh_paths(p.join(self.asset_dir, self._glob_pattern))

    def _cache_asset(self, asset_id):
        r"""Actually cache this asset_id.
        """
        mesh = mio.import_mesh(self.asset_paths[asset_id])
        if isinstance(mesh, TexturedTriMesh):
            self._cache_image_for_id(asset_id, mesh.texture)
        self._cache_mesh_for_id(asset_id, mesh)

    def _cache_mesh_for_id(self, asset_id, mesh):
        asset_cache_dir = p.join(self.cache_dir, asset_id)
        mesh_path = p.join(asset_cache_dir, MESH_FILENAME)
        with gzip.open(mesh_path, 'wb') as f:
            json.dump(mesh.tojson(), f)

    def mesh_json(self, asset_id):
        return p.join(safe_join(self.cache_dir, asset_id), MESH_FILENAME)
