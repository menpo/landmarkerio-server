import abc
from flask import Flask, request, send_file
from flask.ext.restful import abort, Api, Resource
from flask.ext.restful.utils import cors


class LandmarkerIOAdapter(object):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API.
    """
    __metaclass__ = abc.ABCMeta

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
                                       headers=['Content-Type'])]

    class Mesh(Resource):

        def get(self, mesh_id):
            try:
                return adapter.mesh_json(mesh_id)
            except:
                abort(404, message="{} is not an available model".format(mesh_id))

    class MeshList(Resource):

        def get(self):
            return adapter.mesh_ids()

    class Texture(Resource):

        def get(self, mesh_id):
            try:
                return send_file(adapter.texture_file(mesh_id),
                                 mimetype='image/jpeg')
            except:
                abort(404, message="{} is not a textured mesh".format(mesh_id))

    class TextureList(Resource):

        def get(self):
            return adapter.textured_mesh_ids()

    class Landmark(Resource):

        def get(self, mesh_id, lm_id):
            try:
                return adapter.landmark_json(mesh_id, lm_id)
            except:
                abort(404, message="{}:{} does not exist".format(mesh_id, lm_id))

        def put(self, mesh_id, lm_id):
            return adapter.save_landmark_json(mesh_id, lm_id, request.json)

    class LandmarkList(Resource):

        def get(self):
            print 'asked for list'
            return adapter.all_landmarks()

    class LandmarkListForId(Resource):

        def get(self, mesh_id):
            return adapter.landmark_ids(mesh_id)

    class Template(Resource):

        def get(self, lm_id):
            try:
                return adapter.template_json(lm_id)
            except:
                abort(404, message="{} template not exist".format(lm_id))

    class TemplateList(Resource):

        def get(self):
            return adapter.templates()

    api_endpoint = '/api/v1/'

    api.add_resource(MeshList, api_endpoint + 'meshes')
    api.add_resource(Mesh, api_endpoint + 'meshes/<string:mesh_id>')

    api.add_resource(TextureList, api_endpoint + 'textures')
    api.add_resource(Texture, api_endpoint + 'textures/<string:mesh_id>')

    api.add_resource(LandmarkList, api_endpoint + 'landmarks')
    api.add_resource(LandmarkListForId, api_endpoint +
                     'landmarks/<string:mesh_id>')
    api.add_resource(Landmark, api_endpoint +
                     'landmarks/<string:mesh_id>/<string:lm_id>')

    api.add_resource(TemplateList, api_endpoint + 'templates')
    api.add_resource(Template, api_endpoint + 'templates/<string:lm_id>')

    return app
