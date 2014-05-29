import abc
from copy import deepcopy
from collections import defaultdict
import glob
import json
import os
import os.path as p
import StringIO

import menpo.io as mio
from menpo.shape.mesh import TexturedTriMesh
import menpo

from .utils import load_template
from .api import (MeshLandmarkerIOAdapter, LandmarkerIOAdapter,
                  ImageLandmarkerIOAdapter)


def as_jpg_file(image):
    p = image.as_PILImage()
    output = StringIO.StringIO()
    p.save(output, format='jpeg')
    output.seek(0)
    return output


def as_jpg_thumbnail_file(img, width=640):
    ip = img.as_PILImage()
    w, h = ip.size
    h2w = h * 1. / w
    ips = ip.resize((width, int(h2w * width)))
    output = StringIO.StringIO()
    ips.save(output, quality=20, format='jpeg')
    output.seek(0)
    return output

blank_tnail = menpo.image.Image.blank((16, 16), n_channels=3)


class MenpoAdapter(LandmarkerIOAdapter):

    def __init__(self, landmark_dir, template_dir=None):
        self.landmark_dir = p.abspath(p.expanduser(landmark_dir))
        if not p.isdir(self.landmark_dir):
            print("Warning the landmark dir does not exist - creating...")
            os.mkdir(self.landmark_dir)
        if template_dir is None:
            # try the user folder
            user_templates = p.expanduser(p.join('~', '.lmiotemplates'))
            if p.isdir(user_templates):
                template_dir = user_templates
            else:
                raise ValueError("No template dir provided and "
                                 "{} doesn't exist".format(user_templates))
        self.template_dir = p.abspath(p.expanduser(template_dir))
        print ('landmarks: {}'.format(landmark_dir))
        print ('templates: {}'.format(template_dir))

    @abc.abstractproperty
    def n_dims(self):
        pass

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
        template_paths = glob.glob(p.join(self.template_dir, '*.txt'))
        print self.template_dir
        print template_paths
        return [p.splitext(p.split(t)[-1])[0] for t in template_paths]

    def template_json(self, lm_id):
        fp = p.join(self.template_dir, lm_id + '.txt')
        return load_template(fp, self.n_dims)


class MeshMenpoAdapter(MenpoAdapter, MeshLandmarkerIOAdapter):

    def __init__(self, model_dir, landmark_dir, template_dir=None):
        MenpoAdapter.__init__(self, landmark_dir, template_dir=template_dir)
        self.model_dir = model_dir
        print ('models:    {}'.format(model_dir))

    @property
    def n_dims(self):
        return 3

    def mesh_paths(self):
        return mio.mesh_paths(p.join(self.model_dir, '*'))

    def texture_paths(self):
        return mio.image_paths(p.join(self.model_dir, '*'))

    def mesh_ids(self):
        return [p.splitext(p.split(m)[1])[0] for m in self.mesh_paths()]

    def mesh_json(self, mesh_id):
        mesh_glob = p.join(self.model_dir, mesh_id + '.*')
        return list(mio.import_meshes(mesh_glob))[0].tojson()

    def textured_mesh_ids(self):
        return [p.splitext(p.split(t)[1])[0] for t in self.texture_paths()]

    def texture_file(self, mesh_id):
        img_glob = p.join(self.model_dir, mesh_id + '.*')
        return as_jpg_file(list(mio.import_images(img_glob))[0])

    def thumbnail_file(self, mesh_id):
        img_glob = p.join(self.model_dir, mesh_id + '.*')
        imgs = list(mio.import_images(img_glob))
        if len(imgs) == 0:
            return as_jpg_thumbnail_file(blank_tnail)
        else:
            return as_jpg_thumbnail_file(imgs[0])


class CachingMeshMenpoAdapter(MeshMenpoAdapter):

    def __init__(self, model_dir, landmark_dir, template_dir):
        MeshMenpoAdapter.__init__(self, model_dir, landmark_dir, template_dir)
        print('Caching meshes and textures...')
        self.meshes = {}
        self.textures = {}
        self.thumbnails = {}
        for mesh in mio.import_meshes(p.join(self.model_dir, '*')):
            mesh_id = mesh.ioinfo.filename
            self.meshes[mesh_id] = mesh.tojson()
            if isinstance(mesh, TexturedTriMesh):
                self.textures[mesh_id] = as_jpg_file(mesh.texture)
                self.thumbnails[mesh_id] = as_jpg_thumbnail_file(mesh.texture)
            else:
                self.thumbnails[mesh_id] = as_jpg_thumbnail_file(blank_tnail)
        print(' - {} meshes imported.'.format(len(self.meshes)))
        print(' - {} meshes are textured.'.format(len(self.textures)))

    def mesh_ids(self):
        return list(self.meshes)

    def mesh_json(self, mesh_id):
        return self.meshes[mesh_id]

    def textured_mesh_ids(self):
        return list(self.textures)

    def texture_file(self, mesh_id):
        return deepcopy(self.textures[mesh_id])

    def thumbnail_file(self, mesh_id):
        return deepcopy(self.thumbnails[mesh_id])


class ImageMenpoAdapter(MenpoAdapter, ImageLandmarkerIOAdapter):

    def __init__(self, image_dir, landmark_dir, template_dir=None):
        MenpoAdapter.__init__(self, landmark_dir, template_dir=template_dir)
        self.image_dir = image_dir
        print ('images:    {}'.format(image_dir))

    @property
    def n_dims(self):
        return 2

    def image_paths(self):
        return mio.image_paths(p.join(self.image_dir, '*'))

    def image_ids(self):
        return [p.splitext(p.split(m)[1])[0] for m in self.image_paths()]

    def image_json(self, image_id):
        img_glob = p.join(self.image_dir, image_id + '.*')
        # Import image, send W, H
        img = list(mio.import_images(img_glob))[0]
        return {'width': img.width,
                'height': img.height}

    def texture_file(self, image_id):
        img_glob = p.join(self.image_dir, image_id + '.*')
        return as_jpg_file(list(mio.import_images(img_glob))[0])

    def thumbnail_file(self, image_id):
        img_glob = p.join(self.image_dir, image_id + '.*')
        return as_jpg_thumbnail_file(list(mio.import_images(img_glob))[0])


class CachingImageMenpoAdapter(ImageMenpoAdapter):

    def __init__(self, image_dir, landmark_dir, template_dir=None):
        ImageMenpoAdapter.__init__(self, image_dir, landmark_dir,
                                   template_dir=template_dir)
        print('Caching images and thumbnails...')
        self.images, self.textures, self.thumbnails = {}, {}, {}
        for img in mio.import_images(p.join(self.image_dir, '*')):
            image_id = img.ioinfo.filename
            self.images[image_id] = {'width': img.width,
                                     'height': img.height}
            self.textures[image_id] = as_jpg_file(img)
            self.thumbnails[image_id] = as_jpg_thumbnail_file(img)
        print(' - {} images imported.'.format(len(self.images)))

    def image_ids(self):
        return list(self.images)

    def image_json(self, image_id):
        return self.images[image_id]

    def texture_file(self, image_id):
        return deepcopy(self.textures[image_id])

    def thumbnail_file(self, image_id):
        return deepcopy(self.thumbnails[image_id])
