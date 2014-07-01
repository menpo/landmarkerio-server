from flask import Flask, request, send_file, make_response
from flask.ext.restful import abort, Api, Resource
from flask.ext.restful.utils import cors


def generate_add

def app_for_adapters(adapter, dev=False):
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
                return adapter.load_lm(asset_id, lm_id)
            except Exception as e:
                print(e)
                return abort(404, message="{}:{} does not "
                                          "exist".format(asset_id, lm_id))

        def put(self, asset_id, lm_id):
            try:
                return adapter.save_lm(asset_id, lm_id,
                                                  request.json)
            except Exception as e:
                print(e)
                return abort(409, message="{}:{} unable to "
                                          "save".format(asset_id, lm_id))

        # Need this here to enable CORS put see http://mzl.la/1rCDkWX
        def options(self, asset_id, lm_id):
            pass

    class LandmarkList(Resource):

        def get(self):
            print 'asked for list'
            return adapter.all_lms()

    class LandmarkListForId(Resource):

        def get(self, asset_id):
            return adapter.lm_ids(asset_id)

    class Template(Resource):

        def get(self, lm_id):
            try:
                return adapter.template_json(lm_id)
            except Exception as e:
                print(e)
                return abort(404, message="{} template not "
                                          "exist".format(lm_id))

    class TemplateList(Resource):

        def get(self):
            return adapter.templates()

    class Thumbnail(Resource):

        def get(self, asset_id):
            try:
                return send_file(adapter.thumbnail_file(asset_id),
                                 mimetype='image/jpeg')
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an "
                                          "asset".format(asset_id))


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


def app_for_image_adapter(adapter, dev=False):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------

    adapter: :class:`ImageLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.

    """
    api, app, api_endpoint = app_for_adapters(adapter, dev=dev)

    class Image(Resource):

        def get(self, asset_id):
            try:
                return send_file(adapter.image_info(asset_id),
                                 mimetype='json')
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an available "
                                          "image".format(asset_id))

    class ImageList(Resource):

        def get(self):
            return adapter.asset_ids()

    class Texture(Resource):

        def get(self, asset_id):
            try:
                return send_file(adapter.texture_file(asset_id),
                                 mimetype='image/jpeg')
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an available "
                                          "image".format(asset_id))

    api.add_resource(ImageList, api_endpoint + 'images')
    api.add_resource(Image, api_endpoint + 'images/<string:asset_id>')
    api.add_resource(Texture, api_endpoint + 'textures/<string:asset_id>')

    return app


def app_for_mesh_adapter(mesh_adapter, dev=False):
    r"""
    Generate a Flask App that will serve images, landmarks and templates to
    landmarker.io

    Parameters
    ----------
    adapter: :class:`MeshLandmarkerIOAdapter`
        Concrete implementation of the Image adapter. Will be queried for
        all data to pass to landmarker.io.
    """
    api, app, api_endpoint = app_for_adapters(mesh_adapter, dev=dev)

    class Mesh(Resource):

        def get(self, asset_id):
            try:
                r = make_response(send_file(mesh_adapter.mesh_json(asset_id),
                                            mimetype='application/json'))
                # we know all meshes are served gzipped, inform the client
                r.headers['Content-Encoding'] = 'gzip'
                return r
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an available "
                                          "mesh".format(asset_id))

    class MeshList(Resource):

        def get(self):
            return mesh_adapter.asset_ids()

    class Image(Resource):

        def get(self, asset_id):
            try:
                return send_file(mesh_adapter.image_info(asset_id),
                                 mimetype='json')
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an available "
                                          "image".format(asset_id))

    class Texture(Resource):

        def get(self, asset_id):
            try:
                return send_file(mesh_adapter.texture_file(asset_id),
                                 mimetype='image/jpeg')
            except Exception as e:
                print(e)
                return abort(404, message="{} is not an "
                                          "image".format(asset_id))

    api.add_resource(MeshList, api_endpoint + 'meshes')
    api.add_resource(Mesh, api_endpoint + 'meshes/<string:asset_id>')
    api.add_resource(Image, api_endpoint + 'images/<string:asset_id>')
    api.add_resource(Texture, api_endpoint + 'textures/<string:asset_id>')

    return app
