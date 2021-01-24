import abc
import json
import os
import os.path as p
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from loguru import logger

from landmarkerio import FileExt, LM_DIRNAME
from landmarkerio.types import PathLike


class LandmarkAdapter(abc.ABC):
    @abc.abstractmethod
    def asset_id_to_lm_id(self) -> Dict[str, Sequence[str]]:
        pass

    @abc.abstractmethod
    def landmark_ids(self, asset_id: str) -> Sequence[str]:
        r"""
        Return a list of lm_ids available for a given asset_id
        """
        pass

    @abc.abstractmethod
    def load_landmark(self, asset_id: str, lm_id: str) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def save_landmark(self, asset_id: str, lm_id: str, lm_json: Dict[str, Any]) -> None:
        pass


class FileLmAdapter(LandmarkAdapter):
    r"""
    Concrete implementation of LmAdapter that serves landmarks from the
    local filesystem.
    """

    def load_landmark(self, asset_id: str, lm_id: str) -> Dict[str, Any]:
        fp = self.landmark_path(asset_id, lm_id)
        if not fp.exists():
            raise FileNotFoundError(fp)
        with fp.open("rt") as f:
            lm = json.load(f)
            return lm

    def save_landmark(self, asset_id: str, lm_id: str, lm_json: Dict[str, Any]) -> None:
        r"""
        Persist a given landmark definition to disk.
        """
        fp = self.landmark_path(asset_id, lm_id)
        with fp.open("w") as f:
            json.dump(lm_json, f, sort_keys=True, indent=4, separators=(",", ": "))

    @abc.abstractmethod
    def landmark_path(self, asset_id: str, lm_id: str) -> Path:
        # where a landmark should exist
        pass


class SeparateDirFileLmAdapter(FileLmAdapter):
    def __init__(self, lm_dir: Optional[PathLike]) -> None:
        if lm_dir is None:
            # By default place the landmarks in the cwd
            lm_dir = Path(os.getcwd()) / LM_DIRNAME
        self.lm_dir = Path(p.abspath(p.expanduser(lm_dir)))
        if self.lm_dir.exists():
            logger.warning("The landmark dir does not exist - creating...")
            self.lm_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("landmarks: {}", self.lm_dir)

    def landmark_path(self, asset_id: str, lm_id: str) -> Path:
        # where a landmark should exist
        return self.lm_dir / asset_id / (lm_id + FileExt.lm)

    def landmark_ids(self, asset_id: str) -> Sequence[str]:
        lm_files = self._landmark_paths(asset_id=asset_id)
        return [f.stem for f in lm_files]

    def asset_id_to_lm_id(self) -> Dict[str, Sequence[str]]:
        r"""
        Return a dict mapping asset ID's to landmark IDs that are
        present on this server for that asset.
        """
        lm_files = self._landmark_paths()
        mapping: Dict[str, List[str]] = defaultdict(list)
        for lm_path in lm_files:
            lm_id = lm_path.stem
            asset_id = lm_path.parent.stem
            mapping[asset_id].append(lm_id)
        return cast(Dict[str, Sequence[str]], mapping)

    def _landmark_paths(self, asset_id: Optional[str] = None) -> Sequence[Path]:
        if asset_id is None:
            asset_id = "*"
        asset_path = self.lm_dir / asset_id
        return [
            f
            for f in asset_path.glob("*")
            if f.exists() and f.suffixes[-1] == FileExt.lm
        ]

    def save_landmark(self, asset_id: str, lm_id: str, lm_json: Dict[str, Any]) -> None:
        r"""
        Persist a given landmark definition to disk.
        """
        subject_dir = self.lm_dir / asset_id
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        super().save_landmark(asset_id, lm_id, lm_json)


class InplaceFileLmAdapter(FileLmAdapter):
    def __init__(self, asset_ids_to_paths: Dict[str, Path]) -> None:
        self.ids_to_paths = asset_ids_to_paths
        logger.debug(
            "Landmarks served inplace - found {} asset with landmarks",
            len(self.asset_id_to_lm_id()),
        )

    def landmark_ids(self, asset_id: str) -> Sequence[str]:
        # always the same!
        if asset_id in self.ids_to_paths:
            return ["inplace"]
        else:
            raise ValueError(f"Unable to find landmark IDs for '{asset_id}'")

    def asset_id_to_lm_id(self) -> Dict[str, Sequence[str]]:
        r"""
        Return a dict mapping asset ID's to landmark IDs that are
        present on this server for that asset.
        """
        return {
            aid: ["inplace"]
            for aid in self.ids_to_paths
            if self._lm_path_for_asset_id(aid).exists()
        }

    def landmark_path(self, asset_id: str, lm_id: str) -> Path:
        # note the lm_id is ignored. We just always return the .ljson file.
        return self._lm_path_for_asset_id(asset_id)

    def _lm_path_for_asset_id(self, asset_id: str) -> Path:
        asset_path = Path(self.ids_to_paths[asset_id])
        return asset_path.with_suffix(FileExt.lm)
