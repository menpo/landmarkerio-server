from functools import partial
from flask import Flask, request, send_file, make_response
from flask.ext.restful import abort, Api, Resource
#from flask.ext.restful.utils import cors
import cors  # until twilio/flask-restful/pull/276 is merged, see the package
from landmarkerio import Server, Endpoints, Mimetype

url = lambda *x: '/' + '/'.join(x)
asset = lambda f: partial(f, '<string:asset_id>')


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
gzip_binary_file = partial(binary_file, gzip=True)


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
    origin = Server.origin
    if dev:
        # in development mode, accept CORS from anyone
        origin = '*'
        app.debug = True
    api.decorators = [cors.crossdomain(origin=origin,
                                       headers=['Origin', 'X-Requested-With',
                                                'Content-Type', 'Accept'],
                                       methods=['HEAD', 'GET', 'POST', 'PATCH',
                                                'PUT', 'OPTIONS', 'DELETE'],
                                       credentials=True)]
    return api, app


def add_mode_endpoint(api, mode):

    if mode not in ['image', 'mesh']:
        raise ValueError("Mode can only be 'image' or 'mesh', "
                         "not {}".format(mode))

    class Mode(Resource):

        def get(self):
            return mode

    api.add_resource(Mode, url(Endpoints.mode))


def add_lm_endpoints(api, lm_adapter, template_adapter):
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
            try:
                return lm_adapter.load_lm(asset_id, lm_id)
            except Exception as e:
                try:
                    return template_adapter.load_template(lm_id)
                except Exception as e:
                    return abort(404, message=err)

        def put(self, asset_id, lm_id):
            try:
                return lm_adapter.save_lm(asset_id, lm_id, request.json)
            except Exception as e:
                print(e)
                return abort(409, message="{}:{} unable to "
                                          "save".format(asset_id, lm_id))

        # Need this here to enable CORS put see http://mzl.la/1rCDkWX
        def options(self, asset_id, lm_id):
            pass

    class LandmarkList(Resource):

        def get(self):
            return lm_adapter.asset_id_to_lm_id()

    class LandmarkListForId(Resource):

        def get(self, asset_id):
            return lm_adapter.lm_ids(asset_id)

    lm_url = partial(url, Endpoints.landmarks)
    api.add_resource(LandmarkList, lm_url())
    api.add_resource(LandmarkListForId, asset(lm_url)())
    api.add_resource(Landmark, asset(lm_url)('<string:lm_id>'))


def add_template_endpoints(api, adapter):

    class Template(Resource):

        def get(self, lm_id):
            err = "{} template does not exist".format(lm_id)
            return safe_send(adapter.load_template(lm_id), err)

    class TemplateList(Resource):

        def get(self):
            return adapter.template_ids()

    templates_url = partial(url, Endpoints.templates)
    api.add_resource(TemplateList, templates_url())
    api.add_resource(Template, templates_url('<string:lm_id>'))


def add_collection_endpoints(api, adapter):

    class Collection(Resource):

        def get(self, collection_id):
            err = "{} collection not exist".format(collection_id)
            return safe_send(adapter.collection(collection_id), err)

    class CollectionList(Resource):

        def get(self):
            return adapter.collection_ids()

    collections_url = partial(url, Endpoints.collections)
    api.add_resource(CollectionList, collections_url())
    api.add_resource(Collection, collections_url('<string:collection_id>'))


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

    image_url = partial(url, Endpoints.images)
    texture_url = partial(url, Endpoints.textures)
    thumbnail_url = partial(url, Endpoints.thumbnail)

    api.add_resource(ImageList, image_url())
    api.add_resource(Image, asset(image_url)())

    api.add_resource(Texture, asset(texture_url)())
    api.add_resource(Thumbnail, asset(thumbnail_url)())


def add_mesh_endpoints(api, adapter):

    class Mesh(Resource):

        def get(self, asset_id):
            err = "{} is not an available mesh".format(asset_id)
            return gzip_binary_file(adapter.mesh(asset_id), err)

    class MeshList(Resource):

        def get(self):
            return adapter.asset_ids()

    mesh_url = partial(url, Endpoints.meshes)
    mesh_asset_url = asset(mesh_url)

    api.add_resource(MeshList, mesh_url())
    api.add_resource(Mesh, mesh_asset_url())
