from enum import Enum
from functools import partial
from flask import Flask, request, send_file, make_response
from flask.ext.restful import abort, Api, Resource
from flask.ext.restful.utils import cors
from landmarkerio import LMIO_ORIGIN, LMIO_SERVER_ENDPOINT

class Mimetype(Enum):
    json = 'application/json'
    jpeg = 'image/jpeg'
    binary = 'application/octet-stream'


def safe_send(x, fail_message):
    try:
        return x
    except Exception as e:
        print(e)
        return abort(404, message=fail_message)


def safe_send_file(mimetype, path, fail_message, gzip=False):
    try:
        r = make_response(send_file(path, mimetype=mimetype))
        if gzip:
            r.headers['Content-Encoding'] = 'gzip'
        return r
    except Exception as e:
        print(e)
        return abort(404, message=fail_message)

image_file = partial(safe_send_file, Mimetype.jpeg)
json_file = partial(safe_send_file, Mimetype.json)
gzip_json_file = partial(safe_send_file, Mimetype.json, gzip=True)
binary_file = partial(safe_send_file, Mimetype.binary)


def lmio_api(dev=False):
    r"""
    Generate a Flask App that will serve meshes landmarks and templates to
    landmarker.io

    Parameters
    ----------
    adapter: :class:`LandmarkerIOAdapter`
        Concrete implementation of the LandmarkerIOAdapter. Will be queried for
        all data to pass to landmarker.io.
    dev: `bool`, optional
        If True, listen to anyone for CORS.

    Returns
    -------
    api, app, api_endpoint
    """
    app = Flask(__name__)
    api = Api(app)
    origin = LMIO_ORIGIN
    if dev:
        # in development mode, accept CORS from anyone
        origin = '*'
        app.debug = True
    api.decorators = [cors.crossdomain(origin=origin,
                                       headers=['Origin', 'X-Requested-With',
                                                'Content-Type', 'Accept'],
                                       methods=['HEAD', 'GET', 'POST', 'PATCH',
                                                'PUT', 'OPTIONS', 'DELETE'])]
    return api, app


def add_mode_endpoint(api, mode):

    if mode not in ['image', 'mesh']:
        raise ValueError("Mode can only be 'image' or 'mesh', "
                         "not {}".format(mode))

    class Mode(Resource):

        def get(self):
            return mode

    api.add_resource(Mode, LMIO_SERVER_ENDPOINT + 'mode')


def add_lm_endpoints(api, adapter):
    r"""
    Generate a Flask App that will serve meshes landmarks and templates to
    landmarker.io

    Parameters
    ----------
    adapter: :class:`LandmarkerIOAdapter`
        Concrete implementation of the LandmarkerIOAdapter. Will be queried for
        all data to pass to landmarker.io.
    dev: `bool`, optional
        If True, listen to anyone for CORS.

    Returns
    -------
    api, app, api_endpoint
    """

    class Landmark(Resource):

        def get(self, asset_id, lm_id):
            err = "{} does not have {} landmarks".format(asset_id, lm_id)
            return safe_send(adapter.load_lm(asset_id, lm_id), err)

        def put(self, asset_id, lm_id):
            try:
                return adapter.save_lm(asset_id, lm_id, request.json)
            except Exception as e:
                print(e)
                return abort(409, message="{}:{} unable to "
                                          "save".format(asset_id, lm_id))

        # Need this here to enable CORS put see http://mzl.la/1rCDkWX
        def options(self, asset_id, lm_id):
            pass

    class LandmarkList(Resource):

        def get(self):
            return adapter.asset_id_to_lm_id()

    class LandmarkListForId(Resource):

        def get(self, asset_id):
            return adapter.lm_ids(asset_id)

    api.add_resource(LandmarkList, LMIO_SERVER_ENDPOINT + 'landmarks')
    api.add_resource(LandmarkListForId, LMIO_SERVER_ENDPOINT +
                     'landmarks/<string:asset_id>')
    api.add_resource(Landmark, LMIO_SERVER_ENDPOINT +
                     'landmarks/<string:asset_id>/<string:lm_id>')


def add_template_endpoints(api, adapter):

    class Template(Resource):

        def get(self, lm_id):
            err = "{} template not exist".format(lm_id)
            return safe_send(adapter.template_json(lm_id), err)

    class TemplateList(Resource):

        def get(self):
            return adapter.template_ids()

    api.add_resource(TemplateList, LMIO_SERVER_ENDPOINT + 'templates')
    api.add_resource(Template, LMIO_SERVER_ENDPOINT +
                     'templates/<string:lm_id>')


def add_collection_endpoints(api, adapter):

    class Collection(Resource):

        def get(self, collection_id):
            err = "{} collection not exist".format(collection_id)
            return safe_send(adapter.collection(collection_id), err)

    class CollectionList(Resource):

        def get(self):
            return adapter.collection_ids()

    api.add_resource(CollectionList, LMIO_SERVER_ENDPOINT + 'collections')
    api.add_resource(Collection, LMIO_SERVER_ENDPOINT +
                     'collections/<string:collection_id>')


def add_image_endpoints(api, adapter):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------

    adapter: :class:`ImageLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.

    """
    class Image(Resource):

        def get(self, asset_id):
            err = "{} does not have an image".format(asset_id)
            return json_file(adapter.image_info(asset_id), err)

    class ImageList(Resource):

        def get(self):
            return adapter.asset_ids()

    class Texture(Resource):

        def get(self, asset_id):
            err = "{} does not have a texture".format(asset_id)
            return image_file(adapter.texture_file(asset_id), err)

    class Thumbnail(Resource):

        def get(self, asset_id):
            err = "{} does not have a thumbnail".format(asset_id)
            return image_file(adapter.thumbnail_file(asset_id), err)

    api.add_resource(ImageList, LMIO_SERVER_ENDPOINT + 'images')
    api.add_resource(Image, LMIO_SERVER_ENDPOINT + 'images/<string:asset_id>')

    api.add_resource(Texture, LMIO_SERVER_ENDPOINT +
                     'textures/<string:asset_id>')
    api.add_resource(Thumbnail, LMIO_SERVER_ENDPOINT +
                     'thumbnails/<string:asset_id>')


def add_mesh_endpoints(api, adapter):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------
    adapter: :class:`MeshLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.
    """
    class Mesh(Resource):

        def get(self, asset_id):
            err = "{} is not an available mesh".format(asset_id)
            return gzip_json_file(adapter.mesh_json(asset_id), err)

    class Points(Resource):

        def get(self, asset_id):
            err = "{} does not have any points".format(asset_id)
            return binary_file(adapter.points(asset_id), err)

    class Trilist(Resource):

        def get(self, asset_id):
            err = "{} does not have a trilist".format(asset_id)
            return binary_file(adapter.trilist(asset_id), err)

    class Normals(Resource):

        def get(self, asset_id):
            err = "{} does not have normals".format(asset_id)
            return binary_file(adapter.normals(asset_id), err)

    class Tcoords(Resource):

        def get(self, asset_id):
            err = "{} does not have any tcoords".format(asset_id)
            return binary_file(adapter.tcoords(asset_id), err)

    class MeshList(Resource):

        def get(self):
            return adapter.asset_ids()

    api.add_resource(MeshList, LMIO_SERVER_ENDPOINT + 'meshes')
    api.add_resource(Mesh, LMIO_SERVER_ENDPOINT + 'meshes/<string:asset_id>')
    api.add_resource(Points, LMIO_SERVER_ENDPOINT +
                     'meshes/<string:asset_id>/points')
    api.add_resource(Trilist, LMIO_SERVER_ENDPOINT +
                     'meshes/<string:asset_id>/trilist')
    api.add_resource(Normals, LMIO_SERVER_ENDPOINT +
                     'meshes/<string:asset_id>/normals')
    api.add_resource(Tcoords, LMIO_SERVER_ENDPOINT +
                     'meshes/<string:asset_id>/tcoords')