import os
import os.path as p
from os.path import expanduser, abspath
import shutil
import gzip
import struct
from functools import partial
from pathlib import Path
import time

import menpo
import menpo3d
from menpo.shape.mesh import TexturedTriMesh
import numpy as np

from landmarkerio import CacheFile


# identifier functions


def filename_as_asset_id(fp):
    # simply return the filename as the asset_id
    return Path(fp).stem


def filepath_as_asset_id_under_dir(asset_dir):
    # find the filepath under asset_dir and return the full path as an asset id
    asset_dir = Path(abspath(expanduser(asset_dir)))

    def path_as_asset_id(fp):
        return "__".join(Path(fp).relative_to(asset_dir).parts)

    return path_as_asset_id


def build_asset_mapping(identifier_f, asset_paths_iter):
    asset_mapping = {}
    for path in asset_paths_iter:
        asset_id = identifier_f(path)
        if asset_id in asset_mapping:
            raise RuntimeError(
                "asset_id {} is not unique - links to {} and "
                "{}".format(asset_id, asset_mapping[asset_id], path)
            )
        asset_mapping[asset_id] = path
    return asset_mapping


# ASSET PATH RESOLUTION


def asset_paths(path_f, asset_dir, glob_pattern):
    return path_f(p.join(asset_dir, glob_pattern))


mesh_paths = partial(asset_paths, menpo3d.io.mesh_paths)
image_paths = partial(asset_paths, menpo.io.image_paths)


def glob_pattern(ext_str, recursive):
    file_glob = "*" + ext_str
    if recursive:
        return os.path.join("**", file_glob)
    else:
        return file_glob


def ensure_asset_dir(asset_dir):
    asset_dir = p.abspath(p.expanduser(asset_dir))
    if not p.isdir(asset_dir):
        raise ValueError("{} is not a directory.".format(asset_dir))
    print("assets:    {}".format(asset_dir))
    return asset_dir


# CACHING


def cache_asset(cache_dir, cache_f, path, asset_id):
    r"""
    Caches the info for a given asset id so it can be efficiently
    served in the future.

    Parameters
    ----------
    asset_id : `str`
    The id of the asset that needs to be cached
    """
    asset_cache_dir = p.join(cache_dir, asset_id)
    if not p.isdir(asset_cache_dir):
        os.mkdir(asset_cache_dir)
    cache_f(cache_dir, path, asset_id)


# IMAGE CACHING


def cache_image(cache_dir, path, asset_id):
    r"""Actually cache this asset_id."""
    img = menpo.io.import_image(path)
    _cache_image_for_id(cache_dir, asset_id, img)


def _cache_image_for_id(cache_dir, asset_id, img):
    asset_cache_dir = p.join(cache_dir, asset_id)
    image_info_path = p.join(asset_cache_dir, CacheFile.image)
    texture_path = p.join(asset_cache_dir, CacheFile.texture)
    thumbnail_path = p.join(asset_cache_dir, CacheFile.thumbnail)
    img_path = img.path

    # WebGL only allows textures of maximum dimension 4096
    ratio = 4096.0 / np.array(img.shape)
    if np.any(ratio < 1):
        # the largest axis of the img could be too big for older browsers.
        # Give a warning.
        print(
            "Warning: {} has shape {}. Dims larger than 4096 may have "
            "issues rendering in older browsers.".format(asset_id, img.shape)
        )

    # 2. Save out the image
    if img_path.suffix == ".jpg":
        # Original was a jpg that was suitable, save it
        shutil.copyfile(str(img_path), texture_path)
    else:
        # Original wasn't a jpg or was too big - make it so
        img.as_PILImage().save(texture_path, format="jpeg")
    # 3. Save out the thumbnail
    save_jpg_thumbnail_file(img, thumbnail_path)


def save_jpg_thumbnail_file(img, path, width=640):
    ip = img.as_PILImage()
    w, h = ip.size
    h2w = h * 1.0 / w
    ips = ip.resize((width, int(h2w * width)))
    ips.save(path, quality=20, format="jpeg")


# MESH CACHING


def cache_mesh(cache_dir, path, asset_id):
    mesh = menpo3d.io.import_mesh(path)
    if isinstance(mesh, TexturedTriMesh):
        _cache_image_for_id(cache_dir, asset_id, mesh.texture)
    _cache_mesh_for_id(cache_dir, asset_id, mesh)


def _cache_mesh_for_id(cache_dir, asset_id, mesh):
    asset_cache_dir = p.join(cache_dir, asset_id)
    mesh_tmp_path = p.join(asset_cache_dir, CacheFile.mesh_tmp)
    mesh_path = p.join(asset_cache_dir, CacheFile.mesh)
    # store out the raw file
    _export_raw_mesh(mesh_tmp_path, mesh)
    # compress the raw and remove the uncompressed
    f_in = open(mesh_tmp_path, "rb")
    f_out = gzip.open(mesh_path, "wb", compresslevel=1)
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.unlink(mesh_tmp_path)


def _export_raw_mesh(path, m):
    normals = False  # for now we are just not exporting normals.
    is_textured = hasattr(m, "tcoords")
    with open(str(path), "wb") as f:
        f.write(struct.pack("IIII", m.n_tris, is_textured, normals, False))
        m.points[m.trilist].astype(np.float32).tofile(f)
        if normals:
            m.vertex_normals[m.trilist].astype(np.float32).tofile(f)
        if is_textured:
            m.tcoords.points[m.trilist].astype(np.float32).tofile(f)


def ensure_cache_dir(cache_dir):
    cache_dir = p.abspath(p.expanduser(cache_dir))
    if not p.isdir(cache_dir):
        print("Warning the cache dir does not exist - creating...")
        os.mkdir(cache_dir)
    print("cache:     {}".format(cache_dir))
    return cache_dir


def serial_cacher(cache, path_asset_id):
    for i, (path, asset_id) in enumerate(path_asset_id):
        print("Caching {}/{} - {}".format(i + 1, len(path_asset_id), asset_id))
        cache(path, asset_id)


def parallel_cacher(cache, path_asset_id, n_jobs=-1):
    from joblib import Parallel, delayed

    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(cache)(path, asset_id) for path, asset_id in path_asset_id
    )


def build_cache(
    cacher_f,
    asset_path_f,
    cache_f,
    identifier_f,
    asset_dir,
    cache_dir,
    recursive=False,
    ext=None,
    glob=None,
):

    # 1. Ensure the asset_dir and cache_dir are present.
    asset_dir = ensure_asset_dir(asset_dir)
    cache_dir = ensure_cache_dir(cache_dir)

    if recursive:
        print("assets dir will be searched recursively.")

    if ext is not None:
        ext_str = "." + ext
        print("only assets of type {} will be " "loaded.".format(ext_str))
    else:
        ext_str = ""
    if glob is None:
        # Figure out the glob pattern and save it
        glob_ptn = glob_pattern(ext_str, recursive)
    else:
        glob_ptn = glob

    print('Using glob: "{}"'.format(glob_ptn))

    # Construct a mapping from id's to file paths
    asset_id_to_paths = build_asset_mapping(
        identifier_f, asset_path_f(asset_dir, glob_ptn)
    )

    # Check cache for what needs to be updated
    asset_ids = set(asset_id_to_paths.keys())
    cached = set(os.listdir(cache_dir))
    uncached = asset_ids - cached

    print("{} assets need to be added to " "the cache".format(len(uncached)))
    cache = partial(cache_asset, cache_dir, cache_f)
    path_asset_id = [(asset_id_to_paths[a_id], a_id) for a_id in uncached]
    start = time.time()
    cacher_f(cache, path_asset_id)
    elapsed = time.time() - start
    if len(uncached) > 0:
        print("{} assets cached in {:.0f} seconds".format(len(uncached), elapsed))
    return cache_dir, asset_id_to_paths


build_mesh_serial_cache = partial(build_cache, serial_cacher, mesh_paths, cache_mesh)
build_image_serial_cache = partial(build_cache, serial_cacher, image_paths, cache_image)

build_mesh_parallel_cache = partial(
    build_cache, parallel_cacher, mesh_paths, cache_mesh
)
build_image_parallel_cache = partial(
    build_cache, parallel_cacher, image_paths, cache_image
)


def cache_assets(
    mode, identifier_f, asset_dir, cache_dir, recursive=False, ext=None, glob=None
):
    r""""""
    if mode == "image":
        cache_builder = build_image_parallel_cache
    elif mode == "mesh":
        cache_builder = build_mesh_parallel_cache
    else:
        raise ValueError("mode must be 'image' or 'mesh'")
    return cache_builder(
        identifier_f, asset_dir, cache_dir, recursive=recursive, ext=ext, glob=glob
    )
