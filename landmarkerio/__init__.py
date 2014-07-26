from enum import Enum


class FileExt(Enum):
    lm = '.json'
    template = '.txt'
    collection = '.txt'


class CacheFile(Enum):
    texture = 'texture.jpg'
    image = 'image.json'
    thumbnail = 'thumbnail.jpg'
    mesh = 'mesh.json.gz'
    points = 'points.blob'
    trilist = 'trilist.blob'
    normals = 'normals.blob'
    tcoords = 'tcoords.blob'


class Server(Enum):
    origin = 'http://www.landmarker.io'
    endpoint = '/api/v1/'


class Endpoints(Enum):
    mode = 'mode'
    images = 'images'
    collections = 'collections'
    landmarks = 'landmarks'
    meshes = 'meshes'
    templates = 'templates'
    points = 'points'
    trilist = 'trilist'
    tcoords = 'tcoords'
    normals = 'normals'
    texture = 'texture'
    thumbnail = 'thumbnails'


class Mimetype(Enum):
    json = 'application/json'
    jpeg = 'image/jpeg'
    binary = 'application/octet-stream'


LM_DIRNAME = 'lmiolandmarks'
TEMPLATE_DINAME = '.lmiotemplates'

ALL_COLLECTION_ID = 'all'

dirs_in_dir = lambda (path): [p for p in path.iterdir() if p.is_dir()]
