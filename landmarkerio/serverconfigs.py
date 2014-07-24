from landmarkerio.server import (lmio_api, add_mode_endpoint, add_lm_endpoints,
                                 add_image_endpoints, add_mesh_endpoints,
                                 add_template_endpoints,
                                 add_collection_endpoints)
from landmarkerio.template import FileTemplateAdapter
from landmarkerio.landmark import FileLmAdapter
from landmarkerio.collection import (AllCacheCollectionAdapter,
                                     FileCollectionAdapter)
from landmarkerio.asset import ImageCacheAdapter, MeshCacheAdapter


def serve_from_cache(mode, cache_dir, lm_dir, template_dir=None,
                     collection_dir=None, dev=False):
    r"""

    """
    api, app = lmio_api(dev=dev)
    add_lm_endpoints(api, FileLmAdapter(lm_dir))
    if mode == 'image':
        n_dims = 2
        endpoint_adder = add_image_endpoints
        asset_adapter = ImageCacheAdapter
    elif mode == 'mesh':
        n_dims = 3
        endpoint_adder = add_mesh_endpoints
        asset_adapter = MeshCacheAdapter
    else:
        raise ValueError("mode must be 'image' or 'mesh'")
    endpoint_adder(api, asset_adapter(cache_dir))
    add_mode_endpoint(api, mode)
    template_adapter = FileTemplateAdapter(n_dims, template_dir=template_dir)
    add_template_endpoints(api, template_adapter)
    if collection_dir is not None:
        collection_adapter = FileCollectionAdapter(collection_dir)
    else:
        collection_adapter = AllCacheCollectionAdapter(cache_dir)
    add_collection_endpoints(api, collection_adapter)
    print collection_adapter
    return app
