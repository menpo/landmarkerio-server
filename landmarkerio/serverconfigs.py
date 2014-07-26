from landmarkerio.server import (lmio_api, add_mode_endpoint, add_lm_endpoints,
                                 add_image_endpoints, add_mesh_endpoints,
                                 add_template_endpoints,
                                 add_collection_endpoints)
from landmarkerio.template import FileTemplateAdapter
from landmarkerio.landmark import FileLmAdapter
from landmarkerio.collection import (AllCacheCollectionAdapter,
                                     FileCollectionAdapter)
from landmarkerio.asset import ImageCacheAdapter, MeshCacheAdapter

from landmarkerio import LMIOServer


def serve_with_cherrypy(app, port=5000):
    import cherrypy
    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app, LMIOServer.endpoint)
    cherrypy.config.update({
        'server.socket_port': port
    })
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


def serve_from_cache(mode, cache_dir, lm_dir, template_dir=None,
                     collection_dir=None, dev=False):
    r"""

    """
    api, app = lmio_api(dev=dev)
    if dev:
        app.debug = True
    add_lm_endpoints(api, FileLmAdapter(lm_dir))
    # always serve at least images
    add_image_endpoints(api, ImageCacheAdapter(cache_dir))
    if mode == 'image':
        n_dims = 2
    elif mode == 'mesh':
        n_dims = 3
        add_mesh_endpoints(api, MeshCacheAdapter(cache_dir))
    else:
        raise ValueError("mode must be 'image' or 'mesh'")
    add_mode_endpoint(api, mode)
    template_adapter = FileTemplateAdapter(n_dims, template_dir=template_dir)
    add_template_endpoints(api, template_adapter)
    if collection_dir is not None:
        collection_adapter = FileCollectionAdapter(collection_dir)
    else:
        collection_adapter = AllCacheCollectionAdapter(cache_dir)
    add_collection_endpoints(api, collection_adapter)
    print(collection_adapter)
    return app
