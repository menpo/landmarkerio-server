import asyncio
import copy
import functools
from typing import Optional, Union

from loguru import logger
from sanic import Blueprint
from sanic.exceptions import SanicException
from sanic.response import json

from landmarkerio.asset import ImageAdapter, MeshAdapter
from landmarkerio.collection import CollectionAdapter, MissingCollection
from landmarkerio.fit import (
    AutoFit,
    Busy,
    FitRegistry,
    NothingToLearn,
    NothingToRefine,
    NotReady,
    UnknownModel,
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
    fit: Optional[Union[AutoFit, FitRegistry]] = None,
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
        registry = FitRegistry.of(fit)

        def model_or_404(name: Optional[str]) -> AutoFit:
            try:
                return registry.get(name)
            except UnknownModel as e:
                raise SanicException(str(e.args[0]), status_code=404)

        def asset_or_404(asset_id: str) -> None:
            if asset_id not in registry.ids_to_paths:
                raise SanicException(f"Unable to find {asset_id}", status_code=404)

        # (errors are JSON whatever the handler: it hands over to the helpers above)
        fit_route = functools.partial(api.route, error_format="json")

        # Every route is for one model: /fit/models/<model>/..., or, without a model, the
        # default one (so a client that knows only one model keeps working).

        async def do_fit(request, model, asset_id):
            r"""The landmarks that are on the image (the body, as the client has them, saved or
            not), refined by the model. Nothing is saved."""
            model_fit = model_or_404(model)
            asset_or_404(asset_id)
            try:
                points = await asyncio.get_running_loop().run_in_executor(
                    None,
                    model_fit.fit,
                    asset_id,
                    (request.json or {}).get("landmarks", {}).get("points"),
                    (request.json or {}).get("box"),
                )
            except (NotReady, NothingToRefine) as e:
                raise SanicException(str(e), status_code=409)
            data = copy.deepcopy(request.json)
            data["landmarks"]["points"] = points.tolist()
            data["fitted_by"] = model_fit.name
            return json(data)

        async def do_box(model, asset_id):
            r"""The box the model would start from, ``[[row_min, col_min], [row_max, col_max]]``."""
            model_fit = model_or_404(model)
            asset_or_404(asset_id)
            try:
                box = await asyncio.get_running_loop().run_in_executor(
                    None, model_fit.suggest_box, asset_id
                )
            except (NotReady, NothingToRefine) as e:
                raise SanicException(str(e), status_code=409)
            return json(box.tolist())

        def do_train(model):
            r"""Learn from all the annotations again, as a background job (see GET /fit)."""
            try:
                return json(model_or_404(model).train())
            except (Busy, NothingToLearn) as e:
                raise SanicException(str(e), status_code=409)

        def do_learn(model, asset_id):
            r"""Add one saved image to the model, as a background job (see GET /fit)."""
            model_fit = model_or_404(model)
            asset_or_404(asset_id)
            try:
                return json(model_fit.learn(asset_id))
            except (Busy, NothingToLearn) as e:
                raise SanicException(str(e), status_code=409)

        def do_learn_on_save(request, model):
            r"""``{"enabled": true/false}``: learn from each image as it is saved, or not."""
            model_fit = model_or_404(model)
            enabled = (request.json or {}).get("enabled")
            if not isinstance(enabled, bool):
                raise SanicException(
                    'expected {"enabled": true or false}', status_code=400
                )
            model_fit.set_learn_on_save(enabled)
            return json(model_fit.status())

        def do_predict(request, model, t_id):
            r"""The pre-fit: fit every image without saved landmarks and keep the result as a
            proposal (served as the landmarks until saved) in the layout of the template, as a
            background job (see GET /fit). The body may have ``overwrite`` and ``chain``."""
            model_fit = model_or_404(model)
            try:
                skeleton = template_adapter.load_template(t_id)
            except MissingTemplate:
                raise SanicException(f"No template {t_id}", status_code=404)
            options = request.json or {}
            try:
                return json(
                    model_fit.predict(
                        skeleton,
                        overwrite=bool(options.get("overwrite")),
                        chain=bool(options.get("chain")),
                    )
                )
            except (Busy, NotReady, NothingToLearn) as e:
                raise SanicException(str(e), status_code=409)

        @fit_route("/fit")
        async def fit_status(request):
            r"""The default model's status, and ``models``: the status of each model by name."""
            return json(registry.status())

        # the default model
        @fit_route("/fit/<asset_id>/<lm_id>", methods=("POST",))
        async def fit_asset(request, asset_id, lm_id):
            return await do_fit(request, None, asset_id)

        @fit_route("/fit/box/<asset_id>")
        async def fit_box(request, asset_id):
            return await do_box(None, asset_id)

        @fit_route("/fit/train", methods=("POST",))
        async def fit_train(request):
            return do_train(None)

        @fit_route("/fit/learn/<asset_id>", methods=("POST",))
        async def fit_learn(request, asset_id):
            return do_learn(None, asset_id)

        @fit_route("/fit/learn_on_save", methods=("POST",))
        async def fit_learn_on_save(request):
            return do_learn_on_save(request, None)

        @fit_route("/fit/predict/<t_id>", methods=("POST",))
        async def fit_predict(request, t_id):
            return do_predict(request, None, t_id)

        # a model by name
        @fit_route("/fit/models/<model>/fit/<asset_id>/<lm_id>", methods=("POST",))
        async def model_fit_asset(request, model, asset_id, lm_id):
            return await do_fit(request, model, asset_id)

        @fit_route("/fit/models/<model>/box/<asset_id>")
        async def model_fit_box(request, model, asset_id):
            return await do_box(model, asset_id)

        @fit_route("/fit/models/<model>/train", methods=("POST",))
        async def model_fit_train(request, model):
            return do_train(model)

        @fit_route("/fit/models/<model>/learn/<asset_id>", methods=("POST",))
        async def model_fit_learn(request, model, asset_id):
            return do_learn(model, asset_id)

        @fit_route("/fit/models/<model>/learn_on_save", methods=("POST",))
        async def model_fit_learn_on_save(request, model):
            return do_learn_on_save(request, model)

        @fit_route("/fit/models/<model>/predict/<t_id>", methods=("POST",))
        async def model_fit_predict(request, model, t_id):
            return do_predict(request, model, t_id)

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
