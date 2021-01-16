import abc
import os
from pathlib import Path
from typing import Sequence

from landmarkerio import CacheFile
from landmarkerio.types import PathLike


class ImageAdapter(abc.ABC):
    @abc.abstractmethod
    def texture_path(self, asset_id: str) -> Path:
        pass

    @abc.abstractmethod
    def thumbnail_path(self, asset_id: str) -> Path:
        pass

    @abc.abstractmethod
    def asset_ids(self) -> Sequence[str]:
        pass


class MeshAdapter(abc.ABC):
    @abc.abstractmethod
    def asset_ids(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def mesh_path(self, asset_id: str) -> Path:
        pass


class CacheAdapter:
    def __init__(self, cache_dir: PathLike) -> None:
        self.cache_dir = Path(os.path.abspath(os.path.expanduser(cache_dir)))


class ImageCacheAdapter(CacheAdapter, ImageAdapter):
    def __init__(self, cache_dir):
        CacheAdapter.__init__(self, cache_dir)
        self._image_asset_ids = [
            a.parent.name
            for a in self.cache_dir.glob(os.path.join("*", CacheFile.image))
            if a.parent.parent == self.cache_dir
        ]

    def texture_path(self, asset_id: str) -> Path:
        return self.cache_dir / asset_id / CacheFile.texture

    def thumbnail_path(self, asset_id: str) -> Path:
        return self.cache_dir / asset_id / CacheFile.thumbnail

    def asset_ids(self) -> Sequence[str]:
        return self._image_asset_ids


class MeshCacheAdapter(CacheAdapter, MeshAdapter):
    def __init__(self, cache_dir: PathLike) -> None:
        CacheAdapter.__init__(self, cache_dir)
        self._mesh_asset_ids = [
            a.parent.name
            for a in self.cache_dir.glob(os.path.join("*", CacheFile.mesh))
            if a.parent.parent == self.cache_dir
        ]

    def mesh_path(self, asset_id: str) -> Path:
        return self.cache_dir / asset_id / CacheFile.mesh

    def asset_ids(self) -> Sequence[str]:
        return self._mesh_asset_ids
