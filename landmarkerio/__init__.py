LMIO_ORIGIN = 'http://www.landmarker.io'
LMIO_SERVER_ENDPOINT = '/api/v1/'

CACHE_DIRNAME = 'lmiocache'
LM_DIRNAME = 'lmiolandmarks'
TEMPLATE_DINAME = '.lmiotemplates'
COLLECTION_DIRNAME = '.lmiocollections'

LM_EXT = '.json'
TEMPLATE_EXT = '.txt'
COLLECTION_EXT = '.txt'
ALL_COLLECTION_ID = 'all'


TEXTURE_FILENAME = 'texture.jpg'
IMAGE_INFO_FILENAME = 'image.json'
THUMBNAIL_FILENAME = 'thumbnail.jpg'
MESH_FILENAME = 'mesh.json.gz'

POINTS_FILENAME = 'points.blob'
TRILIST_FILENAME = 'trilist.blob'
NORMALS_FILENAME = 'normals.blob'
TCOORDS_FILENAME = 'tcoords.blob'

dirs_in_dir = lambda (path): [p for p in path.iterdir() if p.is_dir()]
