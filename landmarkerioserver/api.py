import abc
from flask import Flask, request, send_file
from flask.ext.restful import abort, Api, Resource
from flask.ext.restful.utils import cors


class LandmarkerIOAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API.

    Note that this implementation is incomplete, as it does include the ability
    to host assets.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def all_landmarks(self):
        pass

    @abc.abstractmethod
    def landmark_ids(self, mesh_id):
        pass

    @abc.abstractmethod
    def landmark_json(self, mesh_id, lm_id):
        pass

    @abc.abstractmethod
    def save_landmark_json(self, mesh_id, lm_id, lm_json):
        pass

    @abc.abstractmethod
    def templates(self):
        pass

    @abc.abstractmethod
    def template_json(self, lm_id):
        pass

    @abc.abstractmethod
    def thumbnail_file(self, image_id):
        pass


class MeshLandmarkerIOAdapter(LandmarkerIOAdapter):
    r"""
    Abstract definition of an adapter that serves mesh assets along with
    landmarks and templates.
    """

    @abc.abstractmethod
    def mesh_ids(self):
        pass

    @abc.abstractmethod
    def mesh_json(self, mesh_id):
        pass

    @abc.abstractmethod
    def textured_mesh_ids(self):
        pass

    @abc.abstractmethod
    def texture_file(self, mesh_id):
        pass


class ImageLandmarkerIOAdapter(LandmarkerIOAdapter):
    r"""
    Abstract definition of an adapter that serves image assets along with
    landmarks and templates.
    """

    @abc.abstractmethod
    def image_ids(self):
        pass

    @abc.abstractmethod
    def image_json(self, image_id):
        pass

    @abc.abstractmethod
    def texture_file(self, image_id):
        pass


def app_for_adapter(adapter, gzip=False, dev=False):
    r"""
    Generate a Flask App that will serve meshes landmarks and templates to
    landmarker.io

    Parameters
    ----------

    adapter: :class:`LandmarkerIOAdapter`
        Concrete implementation of the LandmarkerIOAdapter. Will be queried for
        all data to pass to landmarker.io.

    gzip: Boolean, optional
        If True, responses will be gzipped before being sent to the client.
        Higher workload for the server, smaller payload to the client.

        Default: False

    dev: Boolean, optional
        If True, listen to anyone for CORS.

        Default: False

    Returns
    -------

    api, app
    """
    app = Flask(__name__)

    if gzip:
        from flask.ext.compress import Compress
        Compress(app)

    api = Api(app)
    origin = 'http://www.landmarker.io'
    if dev:
        # in development mode, accept CORS from anyone
        origin = '*'
    api.decorators = [cors.crossdomain(origin=origin,
                                       headers=['Origin', 'X-Requested-With',
                                                'Content-Type', 'Accept'],
                                       methods=['HEAD', 'GET', 'POST', 'PATCH',
                                                'PUT', 'OPTIONS', 'DELETE'])]

    class Landmark(Resource):

        def get(self, asset_id, lm_id):
            try:
                return adapter.landmark_json(asset_id, lm_id)
            except:
                return abort(404, message="{}:{} does not exist".format(asset_id, lm_id))

        def put(self, asset_id, lm_id):
            try:
                return adapter.save_landmark_json(asset_id, lm_id, request.json)
            except:
                return abort(409, message="{}:{} unable to save".format(asset_id, lm_id))

        # Need this here to enable CORS put! Not sure why...
        def options(self, asset_id, lm_id):
            pass

    class LandmarkList(Resource):

        def get(self):
            print 'asked for list'
            return adapter.all_landmarks()

    class LandmarkListForId(Resource):

        def get(self, asset_id):
            return adapter.landmark_ids(asset_id)

    class Template(Resource):

        def get(self, lm_id):
            try:
                return adapter.template_json(lm_id)
            except:
                return abort(404, message="{} template not exist".format(lm_id))

    class TemplateList(Resource):

        def get(self):
            return adapter.templates()

    class Thumbnail(Resource):

        def get(self, asset_id):
            try:
                return send_file(adapter.thumbnail_file(asset_id),
                                 mimetype='image/jpeg')
            except:
                return abort(404, message="{} is not an asset".format(asset_id))


    api_endpoint = '/api/v1/'

    api.add_resource(LandmarkList, api_endpoint + 'landmarks')
    api.add_resource(LandmarkListForId, api_endpoint +
                     'landmarks/<string:asset_id>')
    api.add_resource(Landmark, api_endpoint +
                     'landmarks/<string:asset_id>/<string:lm_id>')

    api.add_resource(TemplateList, api_endpoint + 'templates')
    api.add_resource(Template, api_endpoint + 'templates/<string:lm_id>')
    api.add_resource(Thumbnail, api_endpoint + 'thumbnails/<string:asset_id>')

    return api, app, api_endpoint


def app_for_mesh_adapter(adapter, gzip=False, dev=False):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------

    adapter: :class:`MeshLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.

    gzip: Boolean, optional
        If True, responses will be gzipped before being sent to the client.
        Higher workload for the server, smaller payload to the client.

        Default: False

    """
    api, app, api_endpoint = app_for_adapter(adapter, gzip=gzip, dev=dev)

    class Mesh(Resource):

        def get(self, mesh_id):
            try:
                return adapter.mesh_json(mesh_id)
            except:
                return abort(404, message="{} is not an available mesh".format(mesh_id))

    class MeshList(Resource):

        def get(self):
            return adapter.mesh_ids()

    class Texture(Resource):

        def get(self, mesh_id):
            try:
                return send_file(adapter.texture_file(mesh_id),
                                 mimetype='image/jpeg')
            except:
                return abort(404, message="{} is not a textured mesh".format(mesh_id))

    class TextureList(Resource):

        def get(self):
            return adapter.textured_mesh_ids()

    api.add_resource(MeshList, api_endpoint + 'meshes')
    api.add_resource(Mesh, api_endpoint + 'meshes/<string:mesh_id>')

    api.add_resource(TextureList, api_endpoint + 'textures')
    api.add_resource(Texture, api_endpoint + 'textures/<string:mesh_id>')

    return app


def app_for_image_adapter(adapter, gzip=False, dev=False):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------

    adapter: :class:`ImageLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.

    gzip: Boolean, optional
        If True, responses will be gzipped before being sent to the client.
        Higher workload for the server, smaller payload to the client.

        Default: False

    """
    api, app, api_endpoint = app_for_adapter(adapter, gzip=gzip, dev=dev)

    class Image(Resource):

        def get(self, image_id):
            try:
                return adapter.image_json(image_id)
            except:
                return abort(404, message="{} is not an available "
                                          "image".format(image_id))

    class ImageList(Resource):

        def get(self):
            return adapter.image_ids()

    class Texture(Resource):

        def get(self, image_id):
            try:
                return send_file(adapter.texture_file(image_id),
                                 mimetype='image/jpeg')
            except:
                return abort(404, message="{} is not an image".format(image_id))

    api.add_resource(ImageList, api_endpoint + 'images')
    api.add_resource(Image, api_endpoint + 'images/<string:image_id>')
    api.add_resource(Texture, api_endpoint + 'textures/<string:image_id>')

    return app
