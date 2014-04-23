from flask import Flask, request, send_file
from flask.ext.restful import abort, Api, Resource
from flask.ext.restful.utils import cors

from config import adapter, config


app = Flask(__name__)

if config.gzip:
    from flask.ext.compress import Compress
    Compress(app)
    
api = Api(app)
api.decorators = [cors.crossdomain(origin='*',
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


if __name__ == '__main__':
    app.run(debug=True)
