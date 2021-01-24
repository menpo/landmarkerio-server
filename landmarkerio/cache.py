import gzip
import os
import os.path as p
import shutil
import struct
import time
from functools import partial
from os.path import abspath, expanduser
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, cast

import menpo
import menpo3d
import numpy as np
from loguru import logger

from landmarkerio import CacheFile
from landmarkerio.mode import UnexpectedMode
from landmarkerio.types import PathLike

PathAssetIDT = Sequence[Tuple[PathLike, str]]
IdentifierF = Callable[[PathLike], str]
CacheF = Callable[[PathLike, str], None]
DirCacheF = Callable[[PathLike, PathLike, str], None]
CacherF = Callable[[CacheF, PathAssetIDT], None]


def filename_as_asset_id(fp: PathLike) -> str:
    # simply return the filename as the asset_id
    return Path(fp).stem


def filepath_as_asset_id_under_dir(asset_dir: PathLike) -> IdentifierF:
    # find the filepath under asset_dir and return the full path as an asset id
    asset_dir = Path(abspath(expanduser(asset_dir)))

    def path_as_asset_id(fp: PathLike) -> str:
        return "__".join(Path(fp).relative_to(asset_dir).parts)

    return path_as_asset_id


def build_asset_mapping(
    identifier_f: IdentifierF, asset_paths_iter: Iterable[PathLike]
) -> Dict[str, Path]:
    asset_mapping: Dict[str, Path] = {}
    for path in asset_paths_iter:
        asset_id = identifier_f(path)
        if asset_id in asset_mapping:
            raise ValueError(
                f"asset_id {asset_id} is not unique - links to {asset_mapping[asset_id]} and {path}"
            )
        asset_mapping[asset_id] = Path(path)

    return asset_mapping


def mesh_paths(asset_dir: PathLike, glob_pattern: str) -> Sequence[Path]:
    return menpo3d.io.mesh_paths(Path(asset_dir) / glob_pattern)


def image_paths(asset_dir: PathLike, glob_pattern: str) -> Sequence[Path]:
    return menpo.io.image_paths(Path(asset_dir) / glob_pattern)


def build_glob_pattern(ext_str: str, recursive: bool) -> str:
    file_glob = f"*{ext_str}"
    if recursive:
        return os.path.join("**", file_glob)
    else:
        return file_glob


def ensure_asset_dir(asset_dir: PathLike) -> Path:
    asset_dir = Path(p.abspath(p.expanduser(asset_dir)))
    if not asset_dir.is_dir():
        raise ValueError(f"{asset_dir} is not a directory")
    logger.debug("assets:    {}", asset_dir)
    return asset_dir


def cache_asset(
    cache_dir: PathLike,
    cache_f: Callable[[PathLike, PathLike, str], None],
    path: PathLike,
    asset_id: str,
) -> None:
    asset_cache_dir = Path(cache_dir) / asset_id
    asset_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_f(cache_dir, path, asset_id)


def cache_image(cache_dir: PathLike, path: PathLike, asset_id: str) -> None:
    r"""Actually cache this asset_id."""
    img = menpo.io.import_image(path)
    _cache_image_for_id(cache_dir, asset_id, img)


def _cache_image_for_id(
    cache_dir: PathLike, asset_id: str, img: menpo.image.Image
) -> None:
    asset_cache_dir = Path(cache_dir) / asset_id
    texture_path = asset_cache_dir / CacheFile.texture
    thumbnail_path = asset_cache_dir / CacheFile.thumbnail
    img_path: Path = getattr(img, "path")

    # WebGL only allows textures of maximum dimension 4096
    ratio = 4096.0 / np.array(img.shape)
    if np.any(ratio < 1):
        # the largest axis of the img could be too big for older browsers.
        # Give a warning.
        logger.warning(
            "Warning: {} has shape {}. Dims larger than 4096 may have "
            "issues rendering in older browsers.",
            asset_id,
            img.shape,
        )

    # 2. Save out the image
    if img_path.suffix == ".jpg":
        # Original was a jpg that was suitable, save it
        shutil.copyfile(img_path, texture_path)
    else:
        # Original wasn't a jpg or was too big - make it so
        menpo.io.export_image(img, texture_path)

    # 3. Save out the thumbnail
    save_jpg_thumbnail_file(img, thumbnail_path)


def save_jpg_thumbnail_file(
    img: menpo.image.Image, path: PathLike, width: int = 640
) -> None:
    ip = img.as_PILImage()
    w, h = ip.size
    h2w = h * 1.0 / w
    ips = ip.resize((width, int(h2w * width)))
    ips.save(path, quality=20, format="jpeg")


def cache_mesh(cache_dir: PathLike, path: PathLike, asset_id: str) -> None:
    mesh = menpo3d.io.import_mesh(path)
    if isinstance(mesh, menpo.shape.mesh.TexturedTriMesh):
        _cache_image_for_id(cache_dir, asset_id, mesh.texture)
    _cache_mesh_for_id(cache_dir, asset_id, mesh)


def _cache_mesh_for_id(
    cache_dir: PathLike, asset_id: str, mesh: menpo.shape.TriMesh
) -> None:
    asset_cache_dir = Path(cache_dir) / asset_id
    mesh_tmp_path = asset_cache_dir / CacheFile.mesh_tmp
    mesh_path = asset_cache_dir / CacheFile.mesh
    _export_raw_mesh(mesh_tmp_path, mesh)

    with mesh_tmp_path.open("rb") as f_in:
        with gzip.open(mesh_path, "wb", compresslevel=1) as f_out:
            f_out.writelines(f_in)

    mesh_tmp_path.unlink()


def _export_raw_mesh(path: PathLike, m: menpo.shape.TriMesh) -> None:
    normals = False  # for now we are just not exporting normals.
    is_textured = hasattr(m, "tcoords")
    with Path(path).open("wb") as f:
        f.write(struct.pack("IIII", m.n_tris, is_textured, normals, False))
        m.points[m.trilist].astype(np.float32).tofile(f)
        if normals:
            m.vertex_normals[m.trilist].astype(np.float32).tofile(f)
        if is_textured:
            m.tcoords.points[m.trilist].astype(np.float32).tofile(f)


def ensure_cache_dir(cache_dir: PathLike) -> Path:
    cache_dir = Path(p.abspath(p.expanduser(cache_dir)))
    if cache_dir.is_dir():
        logger.warning("Warning the cache dir does not exist - creating...")
        cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("cache:     {}", cache_dir)
    return cache_dir


def serial_cacher(cache: CacheF, path_asset_id: PathAssetIDT) -> None:
    for i, (path, asset_id) in enumerate(path_asset_id):
        logger.debug("Caching {}/{} - {}", i + 1, len(path_asset_id), asset_id)
        cache(path, asset_id)


def parallel_cacher(
    cache: CacheF, path_asset_id: PathAssetIDT, n_jobs: int = -1
) -> None:
    from joblib import Parallel, delayed

    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(cache)(path, asset_id) for path, asset_id in path_asset_id
    )


def build_cache(
    cacher_f: CacherF,
    asset_path_f: Callable[[PathLike, str], Sequence[Path]],
    cache_f: DirCacheF,
    identifier_f: IdentifierF,
    asset_dir: PathLike,
    cache_dir: PathLike,
    recursive: bool = False,
    ext: Optional[str] = None,
    glob: Optional[str] = None,
) -> Tuple[Path, Dict[str, Path]]:
    # 1. Ensure the asset_dir and cache_dir are present.
    asset_dir = ensure_asset_dir(asset_dir)
    cache_dir = ensure_cache_dir(cache_dir)

    if recursive:
        logger.debug("assets dir will be searched recursively.")

    if ext is not None:
        ext_str = "." + ext
        logger.debug("only assets of type {} will be " "loaded.", ext_str)
    else:
        ext_str = ""

    if glob is None:
        # Figure out the glob pattern and save it
        glob_ptn = build_glob_pattern(ext_str, recursive)
    else:
        glob_ptn = glob

    logger.debug('Using glob: "{}"', glob_ptn)

    # Construct a mapping from id's to file paths
    asset_id_to_paths = build_asset_mapping(
        identifier_f, asset_path_f(asset_dir, glob_ptn)
    )

    # Check cache for what needs to be updated
    asset_ids = set(asset_id_to_paths.keys())
    cached = set(os.listdir(cache_dir))
    uncached = asset_ids - cached

    logger.debug("{} assets need to be added to " "the cache", len(uncached))
    cache: CacheF = cast(CacheF, partial(cache_asset, cache_dir, cache_f))
    path_asset_id = [(asset_id_to_paths[a_id], a_id) for a_id in uncached]

    start = time.time()
    cacher_f(cache, path_asset_id)
    elapsed = time.time() - start
    if uncached:
        logger.debug("{} assets cached in {:.0f} seconds", len(uncached), elapsed)

    return cache_dir, asset_id_to_paths


def build_image_cache(
    identifier_f: IdentifierF,
    asset_dir: PathLike,
    cache_dir: PathLike,
    recursive: bool = False,
    ext: Optional[str] = None,
    glob: Optional[str] = None,
    parallel: bool = True,
) -> Tuple[Path, Dict[str, Path]]:
    cacher_f: CacherF
    if parallel:
        cacher_f = parallel_cacher
    else:
        cacher_f = serial_cacher

    return build_cache(
        cacher_f=cacher_f,
        asset_path_f=image_paths,
        cache_f=cache_image,
        identifier_f=identifier_f,
        asset_dir=asset_dir,
        cache_dir=cache_dir,
        recursive=recursive,
        ext=ext,
        glob=glob,
    )


def build_mesh_cache(
    identifier_f: IdentifierF,
    asset_dir: PathLike,
    cache_dir: PathLike,
    recursive: bool = False,
    ext: Optional[str] = None,
    glob: Optional[str] = None,
    parallel: bool = True,
) -> Tuple[Path, Dict[str, Path]]:
    cacher_f: CacherF
    if parallel:
        cacher_f = parallel_cacher
    else:
        cacher_f = serial_cacher

    return build_cache(
        cacher_f=cacher_f,
        asset_path_f=mesh_paths,
        cache_f=cache_mesh,
        identifier_f=identifier_f,
        asset_dir=asset_dir,
        cache_dir=cache_dir,
        recursive=recursive,
        ext=ext,
        glob=glob,
    )


def cache_assets(
    mode: str,
    identifier_f: IdentifierF,
    asset_dir: PathLike,
    cache_dir: PathLike,
    recursive: bool = False,
    ext: Optional[str] = None,
    glob: Optional[str] = None,
    parallel: bool = True,
) -> Tuple[Path, Dict[str, Path]]:
    if mode == "image":
        cache_builder = partial(build_image_cache, parallel=parallel)
    elif mode == "mesh":
        cache_builder = partial(build_mesh_cache, parallel=parallel)
    else:
        raise UnexpectedMode(mode)

    return cache_builder(
        identifier_f, asset_dir, cache_dir, recursive=recursive, ext=ext, glob=glob
    )
