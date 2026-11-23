r"""
Learn from the landmarks that were saved, and suggest where the landmarks of an image are.

An :class:`AutoFit` keeps an Active Appearance Model (a patch-based one, which works well on
faces and on most things with a few clear parts) of the images whose landmarks the annotator has
*reviewed*, that is saved through the server (the LJSON then has ``"reviewed": true``). It grows as
the annotation goes on: a newly reviewed image is added to the model incrementally, and the model
is built again when the landmarks of an image it already learnt from are saved once more (an
incremental model cannot forget what it learnt). Learning happens in a background thread, and a
fit uses the latest finished model.

A fit *refines* the landmarks that are on the image: the fitter starts from them, so any starting
point (a detector's, the previous image's) is improved and a good one stays good. It needs
``pip install "landmarkerio-server[fit]"``.
"""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

LJSON_GROUP = "LJSON"


class NotReady(RuntimeError):
    """There is no model yet: too few images have been reviewed."""


class NothingToRefine(ValueError):
    """The image has no landmarks (or not all of them), so there is no starting point."""


class AutoFit:
    r"""
    The model that learns from saved landmarks and fits new images.

    Parameters
    ----------
    ids_to_paths : `dict` of `str` to `Path`
        The image of every asset id. The landmarks are the ``.ljson`` next to it.
    min_images : `int`, optional
        The number of reviewed images before there is a model.
    from_saved : `bool`, optional
        If ``True`` only images that were saved through the server (``"reviewed": true``)
        are learnt from; if ``False`` every image that has complete landmarks is.
    background : `bool`, optional
        Learn in a background thread (``False`` is for tests: it learns before returning).
    """

    def __init__(
        self,
        ids_to_paths: Dict[str, Path],
        min_images: int = 8,
        from_saved: bool = True,
        background: bool = True,
    ) -> None:
        self.ids_to_paths = ids_to_paths
        self.min_images = min_images
        self.from_saved = from_saved
        self._background = background
        self._lock = threading.Lock()  # the model and the fitter
        self._work = threading.Lock()  # one learning step at a time
        self._executor = ThreadPoolExecutor(max_workers=1) if background else None
        self._pending = False
        self._training = False
        self._model: Any = None
        self._fitter: Any = None
        self._learnt: Dict[
            str, float
        ] = {}  # asset id -> modification time of the landmarks learnt
        self._error: Optional[str] = None

    # ---- what there is to learn from
    def _ljson_path(self, asset_id: str) -> Path:
        return Path(self.ids_to_paths[asset_id]).with_suffix(".ljson")

    def _read(self, asset_id: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self._ljson_path(asset_id).read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _complete_points(data: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not data:
            return None
        points = data.get("landmarks", {}).get("points")
        if not points or any(v is None for p in points for v in p):
            return None
        return np.array(points, dtype=float)

    def annotated_ids(self) -> List[str]:
        """The assets that the model learns from."""
        ids = []
        for asset_id in sorted(self.ids_to_paths):
            data = self._read(asset_id)
            if self._complete_points(data) is None:
                continue
            if self.from_saved and not (data or {}).get("reviewed"):
                continue
            ids.append(asset_id)
        return ids

    def status(self) -> Dict[str, Any]:
        with self._lock:
            ready = self._fitter is not None
            learnt = len(self._learnt)
        return {
            "ready": ready,
            "training": self._training,
            "trained_on": learnt,
            "annotated": len(self.annotated_ids()),
            "min_images": self.min_images,
            "from_saved": self.from_saved,
            "error": self._error,
        }

    # ---- learning
    def notify_saved(self, asset_id: str) -> None:
        """Landmarks were saved: learn from them (in the background)."""
        if self._executor is None:
            self.refresh()
            return
        self._pending = True
        self._executor.submit(self._drain)

    def _drain(self) -> None:
        while self._pending:
            self._pending = False
            try:
                self.refresh()
            except Exception as error:  # noqa: BLE001
                self._error = f"{type(error).__name__}: {error}"
                logger.exception("learning from the saved landmarks failed")

    def refresh(self) -> None:
        """Learn from whatever is new: add new images, or build again after a change."""
        with self._work:
            self._training = True
            try:
                ids = self.annotated_ids()
                if len(ids) < self.min_images:
                    return
                mtimes = {i: self._ljson_path(i).stat().st_mtime for i in ids}
                with self._lock:
                    learnt, model = dict(self._learnt), self._model
                new = [i for i in ids if i not in learnt]
                changed = [i for i in learnt if i in mtimes and mtimes[i] > learnt[i]]
                gone = [i for i in learnt if i not in mtimes]
                if model is not None and not (new or changed or gone):
                    return
                if model is None or changed or gone:
                    logger.info("building the model from {} images", len(ids))
                    model = self._build_model([self._load(i) for i in ids])
                    learnt = {}
                else:
                    logger.info("adding {} images to the model", len(new))
                    model = self._increment(model, [self._load(i) for i in new])
                learnt.update({i: mtimes[i] for i in ids})
                fitter = self._make_fitter(model)
                with self._lock:
                    self._model, self._fitter, self._learnt = model, fitter, learnt
                    self._error = None
            finally:
                self._training = False

    # ---- fitting
    def fit(
        self,
        asset_id: str,
        start: Optional[Sequence[Sequence[Optional[float]]]] = None,
        max_iters: Sequence[int] = (15, 10),
    ) -> np.ndarray:
        r"""
        Refine the landmarks of an image.

        Parameters
        ----------
        asset_id : `str`
            The image.
        start : ``(n_points, 2)`` array-like, optional
            Where the landmarks are now (what is on the annotator's screen, saved or not).
            The ``.ljson`` of the image if not given.

        Returns
        -------
        points : ``(n_points, 2)`` `ndarray`
            The new positions, (row, column) like the points of the LJSON.

        Raises
        ------
        NotReady
            No model yet.
        NothingToRefine
            The image has no complete landmarks to start from.
        """
        with self._lock:
            fitter = self._fitter
        if fitter is None:
            raise NotReady(
                f"the model needs {self.min_images} reviewed images "
                f"(save the landmarks of a few more); it has learnt from {len(self._learnt)}"
            )
        if start is None:
            points = self._complete_points(self._read(asset_id))
        else:
            points = self._complete_points({"landmarks": {"points": start}})
        if points is None:
            raise NothingToRefine(
                "the image has no landmarks to refine: place them, or copy the previous ones"
            )
        return self._fit(fitter, asset_id, points, list(max_iters))

    # ---- the parts that need menpofit
    def _load(self, asset_id: str):
        import menpo.io as mio

        image = mio.import_image(self.ids_to_paths[asset_id])
        if image.n_channels == 3:
            image = image.as_greyscale()
        # smaller images are cheaper to learn from, and the fitter works on any size
        image = image.crop_to_landmarks_proportion(0.2, group=LJSON_GROUP)
        diagonal = image.diagonal()
        if diagonal > 400:
            image = image.rescale(400.0 / diagonal)
        return image

    def _build_model(self, images: Sequence[Any]):
        from menpo.feature import igo
        from menpofit.aam import PatchAAM

        return PatchAAM(
            images,
            group=LJSON_GROUP,
            patch_shape=[(15, 15), (23, 23)],
            diagonal=150,
            scales=(0.5, 1.0),
            holistic_features=igo,
            max_shape_components=20,
            max_appearance_components=100,
            verbose=False,
        )

    def _increment(self, model: Any, images: Sequence[Any]):
        # grow a copy: a fit may be using the current model
        grown = copy.deepcopy(model)
        grown.increment(images, group=LJSON_GROUP, verbose=False)
        return grown

    def _make_fitter(self, model: Any):
        from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

        return LucasKanadeAAMFitter(model, lk_algorithm_cls=WibergInverseCompositional)

    def _fit(
        self, fitter: Any, asset_id: str, start: np.ndarray, max_iters: List[int]
    ) -> np.ndarray:
        import menpo.io as mio
        from menpo.shape import PointCloud

        image = mio.import_image(self.ids_to_paths[asset_id])
        if image.n_channels == 3:
            image = image.as_greyscale()
        result = fitter.fit_from_shape(image, PointCloud(start), max_iters=max_iters)
        return np.asarray(result.final_shape.points)


def require_fit_dependencies() -> None:
    """Raise a helpful error if menpofit is not installed."""
    try:
        import menpofit  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Auto-fit needs menpofit: pip install 'landmarkerio-server[fit]'"
        ) from error


# the type of the hooks, for the adapter
SavedHook = Callable[[str], None]
