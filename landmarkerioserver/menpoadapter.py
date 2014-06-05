import abc
from collections import defaultdict
import glob
import json
import os
import os.path as p
import shutil
import gzip

import menpo.io as mio
from menpo.shape.mesh import TexturedTriMesh
import menpo

from .utils import load_template
from .api import (MeshLandmarkerIOAdapter, LandmarkerIOAdapter,
                  ImageLandmarkerIOAdapter)

CACHE_DIRNAME = 'lmiocache'
TEMPLATE_DINAME = '.lmiotemplates'
TEMPLATE_EXT = '.txt'
TEXTURE_FILENAME = 'texture.jpg'
IMAGE_INFO_FILENAME = 'image.json'
THUMBNAIL_FILENAME = 'thumbnail.jpg'
MESH_FILENAME = 'mesh.json.gz'
LANDMARKS_DIR = 'lmiolandmarks'


def asset_id_for_path(fp):
    return p.splitext(p.split(fp)[-1])[0]


def save_jpg_thumbnail_file(img, path, width=640):
    ip = img.as_PILImage()
    w, h = ip.size
    h2w = h * 1. / w
    ips = ip.resize((width, int(h2w * width)))
    ips.save(path, quality=20, format='jpeg')


def autocache(f):
    def wrapped(self, asset_id):
        if not asset_id in self._asset_paths:
            # asset hasn't been cached yet
            self.cache_asset(asset_id)
        f(self, asset_id)
    return wrapped


class MenpoAdapter(LandmarkerIOAdapter):

    def __init__(self, asset_dir, recursive=False, landmark_dir=None,
                 template_dir=None, cache_dir=None):
        # 1. asset dir
        self.asset_dir = p.abspath(p.expanduser(asset_dir))
        if not p.isdir(self.asset_dir):
            raise ValueError('{} is not a directory.'.format(self.asset_dir))
        print ('assets:    {}'.format(self.asset_dir))
        self.recursive = recursive
        if self.recursive:
            print('assets dir will be searched recursively.')
        # 2. landmark dir
        if landmark_dir is None:
            # By default place the landmarks in the cwd
            landmark_dir = p.join(os.getcwd(), LANDMARKS_DIR)
        self.landmark_dir = p.abspath(p.expanduser(landmark_dir))
        if not p.isdir(self.landmark_dir):
            print("Warning the landmark dir does not exist - creating...")
            os.mkdir(self.landmark_dir)

        # 3. template dir
        if template_dir is None:
            # try the user folder
            user_templates = p.expanduser(p.join('~', TEMPLATE_DINAME))
            if p.isdir(user_templates):
                template_dir = user_templates
            else:
                raise ValueError("No template dir provided and "
                                 "{} doesn't exist".format(user_templates))
        self.template_dir = p.abspath(p.expanduser(template_dir))

        # 4. cache dir
        if cache_dir is None:
            # By default place the cache in the cwd
            cache_dir = p.join(os.getcwd(), CACHE_DIRNAME)
        self.cache_dir = p.abspath(p.expanduser(cache_dir))
        if not p.isdir(self.cache_dir):
            print("Warning the cache dir does not exist - creating...")
            os.mkdir(self.cache_dir)
        print ('landmarks: {}'.format(self.landmark_dir))
        print ('templates: {}'.format(self.template_dir))
        print ('cache:     {}'.format(self.cache_dir))

        # Construct a mapping from id's to file paths
        self.asset_paths = {}
        self._rebuild_asset_mapping()

        # Handle aggressive cache at startup
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

    @abc.abstractproperty
    def n_dims(self):
        pass

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
            self._rebuild_asset_mapping()
        if not asset_id in self.asset_paths:
            raise ValueError('{} is not a valid asset_id'.format(asset_id))
        asset_cache_dir = p.join(self.cache_dir, asset_id)
        if not p.isdir(asset_cache_dir):
            print("Cache for {} does not exist - creating...".format(asset_id))
            os.mkdir(asset_cache_dir)
        self._cache_asset(asset_id)

    @property
    def _glob_pattern(self):
        if self.recursive:
            return os.path.join('**', '*')
        else:
            return '*'

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

    def _rebuild_asset_mapping(self):
        self.asset_paths = {}
        for path in self._asset_paths():
            asset_id = asset_id_for_path(path)
            if asset_id in self.asset_paths:
                raise RuntimeError(
                    "asset_id {} is not unique - links to {} and "
                    "{}".format(asset_id, self.asset_paths[asset_id], path))
            self.asset_paths[asset_id] = path

    def asset_ids(self):
        # whenever a client requests the ids freshen the list up
        self._rebuild_asset_mapping()
        return self.asset_paths.keys()

    def landmark_fp(self, asset_id, lm_id):
        return p.join(self.landmark_dir, asset_id, lm_id + '.json')

    def landmark_paths(self, asset_id=None):
        if asset_id is None:
            asset_id = '*'
        g = glob.glob(p.join(self.landmark_dir, asset_id, '*'))
        return filter(lambda f: p.isfile(f) and
                                p.splitext(f)[-1] == '.json', g)

    def all_landmarks(self):
        landmark_files = self.landmark_paths()
        mapping = defaultdict(list)
        for lm_path in landmark_files:
            dir_path, filename = p.split(lm_path)
            lm_set = p.splitext(filename)[0]
            lm_id = p.split(dir_path)[1]
            mapping[lm_id].append(lm_set)
        return mapping

    def landmark_ids(self, asset_id):
        landmark_files = self.landmark_paths(asset_id=asset_id)
        return [p.splitext(p.split(f)[-1])[0] for f in landmark_files]

    def landmark_json(self, asset_id, lm_id):
        fp = self.landmark_fp(asset_id, lm_id)
        if not p.isfile(fp):
            raise IOError
        with open(fp, 'rb') as f:
            lm = json.load(f)
            return lm

    def save_landmark_json(self, asset_id, lm_id, lm_json):
        subject_dir = p.join(self.landmark_dir, asset_id)
        if not p.isdir(subject_dir):
            os.mkdir(subject_dir)
        fp = self.landmark_fp(asset_id, lm_id)
        with open(fp, 'wb') as f:
            json.dump(lm_json, f, sort_keys=True, indent=4,
                      separators=(',', ': '))

    def templates(self):
        template_paths = glob.glob(p.join(self.template_dir,
                                          '*' + TEMPLATE_EXT))
        print self.template_dir
        print template_paths
        return [p.splitext(p.split(t)[-1])[0] for t in template_paths]

    def template_json(self, lm_id):
        fp = p.join(self.template_dir, lm_id + TEMPLATE_EXT)
        return load_template(fp, self.n_dims)


class ImageMenpoAdapter(MenpoAdapter, ImageLandmarkerIOAdapter):

    @property
    def n_dims(self):
        return 2

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

    @autocache
    def image_info(self, asset_id):
        info_path = p.join(self.cache_dir, asset_id, IMAGE_INFO_FILENAME)
        if asset_id in self.asset_paths and p.isfile(info_path):
            return info_path
        else:
            raise ValueError

    @autocache
    def texture_file(self, asset_id):
        texture_path = p.join(self.cache_dir, asset_id, TEXTURE_FILENAME)
        if asset_id in self.asset_paths and p.isfile(texture_path):
            return texture_path
        else:
            raise ValueError

    @autocache
    def thumbnail_file(self, asset_id):
        thumbnail_path = p.join(self.cache_dir, asset_id, THUMBNAIL_FILENAME)
        print('{} - thumbnail_file: {}'.format(asset_id, thumbnail_path))
        if asset_id in self.asset_paths and p.isfile(thumbnail_path):
            return thumbnail_path
        else:
            raise ValueError


class MeshMenpoAdapter(ImageMenpoAdapter, MeshLandmarkerIOAdapter):

    def __init__(self, asset_dir, recursive=False, landmark_dir=None,
                 template_dir=None, cache_dir=None):
        ImageMenpoAdapter.__init__(self, asset_dir,
                                   recursive=recursive,
                                   landmark_dir=landmark_dir,
                                   template_dir=template_dir,
                                   cache_dir=cache_dir)
        self.meshes = {}

    @property
    def n_dims(self):
        return 3

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

    @autocache
    def mesh_json(self, asset_id):
        mesh_path = p.join(self.cache_dir, asset_id, MESH_FILENAME)
        print('{} - mesh_json: {}'.format(asset_id, mesh_path))
        if asset_id in self.asset_paths and p.isfile(mesh_path):
            return mesh_path
        else:
            raise ValueError
