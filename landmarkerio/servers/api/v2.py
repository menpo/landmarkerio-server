from loguru import logger
from sanic import Blueprint
from sanic.exceptions import SanicException
from sanic.response import json

from landmarkerio.asset import ImageAdapter, MeshAdapter
from landmarkerio.collection import CollectionAdapter, MissingCollection
from landmarkerio.landmark import LandmarkAdapter
from landmarkerio.response import serve_gzip_binary_file, serve_image_file
from landmarkerio.template import MissingTemplate, TemplateAdapter


def build_v2_blueprint(
    mode: str,
    collection_adapter: CollectionAdapter,
    template_adapter: TemplateAdapter,
    image_adapter: ImageAdapter,
    mesh_adapter: MeshAdapter,
    landmark_adapter: LandmarkAdapter,
) -> Blueprint:
    api = Blueprint("v2", url_prefix="/api/v2")

    @api.route("/mode")
    async def get_mode(request):
        return json(mode)

    @api.route("/collections")
    async def collections(request):
        return json(collection_adapter.collection_ids())

    @api.route("/collections/<collection_id>")
    async def collection(request, collection_id):
        try:
            return json(collection_adapter.collection(collection_id))
        except MissingCollection as e:
            raise SanicException(str(e), status_code=404)

    @api.route("/templates")
    async def templates(request):
        return json(template_adapter.template_ids())

    @api.route("/templates/<t_id>")
    async def template(request, t_id):
        try:
            return json(template_adapter.load_template(t_id))
        except MissingTemplate as e:
            raise SanicException(str(e), status_code=404)

    @api.route("/images")
    async def images(request):
        return json(image_adapter.asset_ids())

    @api.route("/textures/<asset_id>")
    async def texture(request, asset_id):
        try:
            return await serve_image_file(image_adapter.texture_path(asset_id))
        except FileNotFoundError:
            raise SanicException(
                status_code=404, message=f"Unable to find texture for {asset_id}"
            )

    @api.route("/thumbnails/<asset_id>")
    async def thumbnail(request, asset_id):
        try:
            return await serve_image_file(image_adapter.thumbnail_path(asset_id))
        except FileNotFoundError:
            raise SanicException(
                status_code=404, message=f"Unable to find thumbnail for {asset_id}"
            )

    @api.route("/landmarks")
    async def landmarks(request):
        return json(landmark_adapter.asset_id_to_lm_id())

    @api.route("/landmarks/<asset_id>")
    async def landmarks_subset(request, asset_id):
        try:
            return json(landmark_adapter.landmark_ids(asset_id))
        except ValueError as e:
            raise SanicException(status_code=404, message=str(e))

    @api.route("/landmarks/<asset_id>/<lm_id>")
    async def landmark(request, asset_id, lm_id):
        try:
            return json(landmark_adapter.load_landmark(asset_id, lm_id))
        except BaseException:
            try:
                logger.exception(f"Unable to load landmarks for {asset_id}/{lm_id}")
                return json(template_adapter.load_template(lm_id))
            except MissingTemplate:
                raise SanicException(
                    status_code=404,
                    message=f"{asset_id} does not have {lm_id} landmarks and no valid template found",
                )

    @api.route("/landmarks/<asset_id>/<lm_id>", methods=("PUT",))
    async def upload_landmarks(request, asset_id, lm_id):
        try:
            landmark_adapter.save_landmark(asset_id, lm_id, request.json)
            return json("success")
        except BaseException:
            logger.exception(f"Unable to save landmarks for {asset_id}/{lm_id}")
            raise SanicException(
                status_code=409, message=f"{asset_id}:{lm_id} unable to save"
            )

    @api.route("/meshes")
    async def meshes(request):
        return json(mesh_adapter.asset_ids())

    @api.route("/meshes/<asset_id>")
    async def mesh(request, asset_id):
        try:
            return await serve_gzip_binary_file(mesh_adapter.mesh_path(asset_id))
        except FileNotFoundError:
            raise SanicException(f"Unable to find mesh for {asset_id}", status_code=404)

    return api
