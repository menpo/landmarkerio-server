class FileExt(object):
    lm = ".ljson"
    template = ".yml"
    collection = ".txt"


class CacheFile(object):
    texture = "texture.jpg"
    image = "image.json"
    thumbnail = "thumbnail.jpg"
    mesh_tmp = "mesh.raw.tmp"
    mesh = "mesh.raw.gz"


class Server(object):
    endpoint = "/api/v2/"


class Endpoints(object):
    mode = "mode"
    images = "images"
    collections = "collections"
    landmarks = "landmarks"
    meshes = "meshes"
    templates = "templates"
    points = "points"
    trilist = "trilist"
    tcoords = "tcoords"
    normals = "normals"
    textures = "textures"
    thumbnail = "thumbnails"


class Mimetype(object):
    json = "application/json"
    jpeg = "image/jpeg"
    binary = "application/octet-stream"


LM_DIRNAME = "lmiolandmarks"
TEMPLATE_DINAME = ".lmiotemplates"

ALL_COLLECTION_ID = "all"

dirs_in_dir = lambda path: sorted([p for p in path.iterdir() if p.is_dir()])

try:
    from ._version import __version__
except ImportError:  # a source checkout that was never installed
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__ = _version("landmarkerio-server")
    except PackageNotFoundError:
        __version__ = "unknown"
