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


class LMIOServer(Enum):
    origin = 'http://www.landmarker.io'
    endpoint = '/api/v1/'


LM_DIRNAME = 'lmiolandmarks'
TEMPLATE_DINAME = '.lmiotemplates'

ALL_COLLECTION_ID = 'all'

dirs_in_dir = lambda (path): [p for p in path.iterdir() if p.is_dir()]
