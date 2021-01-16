from typing import Optional

from sanic import Blueprint, Sanic
from sanic_cors import CORS

from landmarkerio.asset import ImageCacheAdapter, MeshCacheAdapter
from landmarkerio.collection import (
    AllCacheCollectionAdapter,
    CollectionAdapter,
    FileCollectionAdapter,
)
from landmarkerio.http_auth.sanic_httpauth import HTTPBasicAuth
from landmarkerio.landmark import LandmarkAdapter
from landmarkerio.servers.api.v2 import build_v2_blueprint
from landmarkerio.servers.auth import verify_password
from landmarkerio.template import CachedFileTemplateAdapter
from landmarkerio.types import PathLike
from landmarkerio.utils import DIMS


def add_basic_auth_to_api(api: Blueprint, username: str, password: str) -> None:
    auth = HTTPBasicAuth()

    @auth.verify_password
    def single_user_verify_password(given_username: str, given_password: str) -> bool:
        return verify_password({username: password}, given_username, given_password)

    @api.middleware("request")
    @auth.login_required
    async def auth_middleware(request):
        pass


def serve_from_cache(
    mode: str,
    cache_dir: PathLike,
    landmark_adapter: LandmarkAdapter,
    template_dir: Optional[PathLike] = None,
    collection_dir: Optional[PathLike] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    app = Sanic(name="landmarkerio")
    CORS(app)

    n_dims = DIMS[mode]
    template_adapter = CachedFileTemplateAdapter(n_dims, template_dir=template_dir)

    collection_adapter: CollectionAdapter
    if collection_dir is not None:
        collection_adapter = FileCollectionAdapter(collection_dir)
    else:
        collection_adapter = AllCacheCollectionAdapter(cache_dir)

    image_adapter = ImageCacheAdapter(cache_dir)
    mesh_adapter = MeshCacheAdapter(cache_dir)

    v2_api = build_v2_blueprint(
        mode,
        collection_adapter,
        template_adapter,
        image_adapter,
        mesh_adapter,
        landmark_adapter,
    )
    app.blueprint(v2_api)

    if username is not None and password is not None:
        add_basic_auth_to_api(v2_api, username, password)

    return app
