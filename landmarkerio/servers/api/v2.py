import asyncio
import copy
from typing import Optional

from loguru import logger
from sanic import Blueprint
from sanic.exceptions import SanicException
from sanic.response import json

from landmarkerio.asset import ImageAdapter, MeshAdapter
from landmarkerio.collection import CollectionAdapter, MissingCollection
from landmarkerio.fit import (
    AutoFit,
    Busy,
    NothingToLearn,
    NothingToRefine,
    NotReady,
)
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
    fit: Optional[AutoFit] = None,
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

    if fit is not None:

        @api.route("/fit")
        async def fit_status(request):
            return json(fit.status())

        @api.route("/fit/<asset_id>/<lm_id>", methods=("POST",))
        async def fit_asset(request, asset_id, lm_id):
            r"""The landmarks that are on the image (the body, as the client has them, saved or
            not), refined by the model that learnt from the saved ones. Nothing is saved."""
            if asset_id not in fit.ids_to_paths:
                raise SanicException(f"Unable to find {asset_id}", status_code=404)
            try:
                points = await asyncio.get_running_loop().run_in_executor(
                    None,
                    fit.fit,
                    asset_id,
                    (request.json or {}).get("landmarks", {}).get("points"),
                    (request.json or {}).get("box"),
                )
            except (NotReady, NothingToRefine) as e:
                raise SanicException(str(e), status_code=409)
            data = copy.deepcopy(request.json)
            data["landmarks"]["points"] = points.tolist()
            return json(data)

        @api.route("/fit/box/<asset_id>")
        async def fit_box(request, asset_id):
            r"""The box the model would start from, ``[[row_min, col_min], [row_max, col_max]]``."""
            if asset_id not in fit.ids_to_paths:
                raise SanicException(f"Unable to find {asset_id}", status_code=404)
            try:
                box = await asyncio.get_running_loop().run_in_executor(
                    None, fit.suggest_box, asset_id
                )
            except (NotReady, NothingToRefine) as e:
                raise SanicException(str(e), status_code=409)
            return json(box.tolist())

        @api.route("/fit/train", methods=("POST",))
        async def fit_train(request):
            r"""Learn from all the annotations again, as a background job (see GET /fit)."""
            try:
                return json(fit.train())
            except (Busy, NothingToLearn) as e:
                raise SanicException(str(e), status_code=409)

        @api.route("/fit/predict/<t_id>", methods=("POST",))
        async def fit_predict(request, t_id):
            r"""Fit every image without landmarks and write them as not reviewed, in the layout
            of the template, as a background job (see GET /fit)."""
            try:
                skeleton = template_adapter.load_template(t_id)
            except MissingTemplate:
                raise SanicException(f"No template {t_id}", status_code=404)
            try:
                return json(fit.predict(skeleton))
            except (Busy, NotReady, NothingToLearn) as e:
                raise SanicException(str(e), status_code=409)

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
