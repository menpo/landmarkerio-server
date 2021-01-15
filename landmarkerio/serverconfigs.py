from sanic import Sanic
from sanic.exceptions import abort
from sanic.response import json
from sanic_cors import CORS

from landmarkerio.asset import ImageCacheAdapter, MeshCacheAdapter
from landmarkerio.collection import AllCacheCollectionAdapter, FileCollectionAdapter
from landmarkerio.server import (
    gzip_binary_file,
    image_file,
    safe_send,
)
from landmarkerio.template import CachedFileTemplateAdapter


def serve_from_cache(
    mode,
    cache_dir,
    lm_adapter,
    template_dir=None,
    upgrade_templates=False,
    collection_dir=None,
    dev=False,
    username=None,
    password=None,
):
    app = Sanic(name="landmarkerio")
    CORS(app)

    if mode == "image":
        n_dims = 2
    elif mode == "mesh":
        n_dims = 3
    else:
        raise ValueError("mode must be 'image' or 'mesh'")

    template_adapter = CachedFileTemplateAdapter(
        n_dims, template_dir=template_dir, upgrade_templates=upgrade_templates
    )

    if collection_dir is not None:
        collection_adapter = FileCollectionAdapter(collection_dir)
    else:
        collection_adapter = AllCacheCollectionAdapter(cache_dir)

    image_adapter = ImageCacheAdapter(cache_dir)
    mesh_adapter = MeshCacheAdapter(cache_dir)

    @app.route("/api/v2/mode")
    async def get_mode(request):
        return json(mode)

    @app.route("/api/v2/collections")
    async def collections(request):
        return json(collection_adapter.collection_ids())

    @app.route("/api/v2/collections/<collection_id>")
    async def collection(request, collection_id):
        err = "{} collection not exist".format(collection_id)
        return json(safe_send(collection_adapter.collection(collection_id), err))

    @app.route("/api/v2/templates")
    async def templates(request):
        return json(template_adapter.template_ids())

    @app.route("/api/v2/collections/<lm_id>")
    async def template(request, lm_id):
        err = "{} template not exist".format(lm_id)
        return json(safe_send(template_adapter.load_template(lm_id), err))

    @app.route("/api/v2/images")
    async def images(request):
        return json(image_adapter.asset_ids())

    @app.route("/api/v2/textures/<asset_id>")
    async def texture(request, asset_id):
        err = "{} does not have a texture".format(asset_id)
        return await image_file(image_adapter.texture_file(asset_id), err)

    @app.route("/api/v2/thumbnails/<asset_id>")
    async def thumbnail(request, asset_id):
        err = "{} does not have a thumbnail".format(asset_id)
        return await image_file(image_adapter.thumbnail_file(asset_id), err)

    @app.route("/api/v2/landmarks")
    async def landmarks(request):
        return lm_adapter.asset_id_to_lm_id()

    @app.route("/api/v2/landmarks/<asset_id>")
    async def landmarks_subset(request, asset_id):
        return lm_adapter.lm_ids(asset_id)

    @app.route("/api/v2/landmarks/<asset_id>/<lm_id>")
    async def landmark(request, asset_id, lm_id):
        err = "{} does not have {} landmarks".format(asset_id, lm_id)
        try:
            return json(lm_adapter.load_lm(asset_id, lm_id))
        except BaseException:
            return json(safe_send(template_adapter.load_template(lm_id), err))

    @app.route("/api/v2/landmarks/<asset_id>/<lm_id>", methods=("PUT",))
    async def upload_landmarks(request, asset_id, lm_id):
        try:
            lm_adapter.save_lm(asset_id, lm_id, request.json)
            return json("success")
        except Exception as e:
            print(e)
            return abort(409, message="{}:{} unable to " "save".format(asset_id, lm_id))

    @app.route("/api/v2/meshes")
    async def meshes(request):
        return json(mesh_adapter.asset_ids())

    @app.route("/api/v2/meshes/<asset_id>")
    async def mesh(request, asset_id):
        err = "{} is not an available mesh".format(asset_id)
        return await gzip_binary_file(mesh_adapter.mesh(asset_id), err)

    return app
