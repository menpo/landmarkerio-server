import abc
from os.path import abspath, expanduser
from pathlib import Path
from typing import Sequence

from loguru import logger
from landmarkerio import ALL_COLLECTION_ID, FileExt, dirs_in_dir
from landmarkerio.types import PathLike


class MissingCollection(ValueError):
    def __init__(self, collection_id: str) -> None:
        super().__init__(f"Cannot find collection with id '{collection_id}'")


def load_collection(path: PathLike) -> Sequence[str]:
    with Path(path).open("rt") as f:
        collection = [line.strip() for line in f.readlines()]
    return [c for c in collection if c]


class CollectionAdapter(abc.ABC):
    @abc.abstractmethod
    def collection_ids(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def collection(self, collection_id: str) -> Sequence[str]:
        pass

    def __str__(self) -> str:
        a = f"Serving {len(self.collection_ids())} collection(s):"
        b = "\n".join(
            f" - {c} ({len(self.collection(c))} assets)" for c in self.collection_ids()
        )
        return f"{a}\n{b}"


class FileCollectionAdapter(CollectionAdapter):
    def __init__(self, collection_dir: PathLike) -> None:
        self.collection_dir = Path(abspath(expanduser(collection_dir)))
        logger.debug("Found collections: {}", self.collection_dir)
        collection_paths = self.collection_dir.glob("*" + FileExt.collection)
        self._collection = {c.stem: load_collection(c) for c in collection_paths}

    def collection_ids(self) -> Sequence[str]:
        return list(self._collection.keys())

    def collection(self, collection_id: str) -> Sequence[str]:
        try:
            return self._collection[collection_id]
        except KeyError:
            raise MissingCollection(collection_id)


class AllCacheCollectionAdapter(CollectionAdapter):
    def __init__(self, cache_dir: PathLike) -> None:
        cache_dir = Path(abspath(expanduser(cache_dir)))
        self._collection = [p.name for p in dirs_in_dir(cache_dir)]
        self._collection_ids = [ALL_COLLECTION_ID]

    def collection_ids(self) -> Sequence[str]:
        return self._collection_ids

    def collection(self, collection_id: str) -> Sequence[str]:
        if collection_id == ALL_COLLECTION_ID:
            return self._collection
        else:
            raise MissingCollection(collection_id)
