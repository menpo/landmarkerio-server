r"""
Learn from the landmarks that were saved, and suggest where the landmarks of an image are.

An :class:`AutoFit` keeps a *model* of the images whose landmarks the annotator has *reviewed*,
that is saved through the server (the LJSON then has ``"reviewed": true``). The model is anything
that follows :class:`FitModel`: it can be built from images, grown by more of them, and made into
something that fits an image from a starting shape. A patch-based AAM is the default (see
:mod:`landmarkerio.fit_models`), and ``lmio --fit-model package.module:Class`` brings your own.

- **Start from a good place.** An initial state (``--fit-init FILE``, whatever
  :meth:`FitModel.load` reads) is the model before it has seen anything: it fits at once, and each
  reviewed image is added to it.
- **Fit without learning.** A fit never changes the model or what is saved; it works as soon as
  there is a model, whether from an initial state or from enough reviewed images.
- **Incremental.** A newly reviewed image is added to a copy of the model in a background thread,
  and a fit uses the latest finished one. When the landmarks of an image the model has already
  learnt from are saved again it starts over (from the initial state if there is one, else from
  scratch), because an incremental model cannot forget.

A fit *refines* the landmarks that are on the image, so any starting point (a detector's, the
previous image's) is improved and a good one stays good.
"""

import abc
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger

LJSON_GROUP = "LJSON"


class NotReady(RuntimeError):
    """There is no model yet: no initial state and too few images have been reviewed."""


class NothingToRefine(ValueError):
    """The image has no landmarks (or not all of them), so there is no starting point."""


class Busy(RuntimeError):
    """A train or predict job is already running."""


class NothingToLearn(ValueError):
    """There is nothing to train on, or the model cannot do what was asked."""


class FitModel(abc.ABC):
    r"""
    What :class:`AutoFit` needs from a model. A *state* is whatever the model is made of (a
    trained AAM, say). The only things a model must do are :meth:`fitter` and :meth:`fit`, so a
    model that was trained elsewhere can be brought as it is. What else it does is optional, and
    what :class:`AutoFit` and the client offer follows from it:

    =================  ===========================================  ==========================
    method             adds                                         without it
    =================  ===========================================  ==========================
    :meth:`increment`  learning from each saved image as you go     no learning on save
    :meth:`build`      training from scratch on all annotations     no (re)training
    :meth:`initialize` fitting an image that has no landmarks yet    needs landmarks to start
    =================  ===========================================  ==========================

    Methods may be called from a background thread, one at a time (a fit can happen at any
    moment though, on the fitter of an earlier state). The images that are given for training
    have their landmarks attached under the group ``LJSON``; :meth:`prepare` loads and conditions
    them.
    """

    #: Training needs at least this many images when there is no initial state.
    min_images = 8

    @abc.abstractmethod
    def fitter(self, state: Any) -> Any:
        r"""Make what :meth:`fit` uses from a state (called once per state, not per fit)."""

    @abc.abstractmethod
    def fit(self, fitter: Any, image: Any, start: np.ndarray) -> np.ndarray:
        r"""
        Fit an image (no landmarks needed) from a starting shape, ``(n_points, 2)`` in
        (row, column) order. Returns the fitted shape, same layout.
        """

    def build(self, images: Sequence[Any]) -> Any:
        r"""Build a state from images (``menpo`` images with an ``LJSON`` landmark group)."""
        raise NotImplementedError

    def increment(self, state: Any, images: Sequence[Any]) -> Any:
        r"""
        Add images to a state and return the new one. The given state may be in use by a fit
        and must not be modified: return a new state (copy it first if updating in place).
        """
        raise NotImplementedError

    def initialize(
        self, state: Any, image: Any, box: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""
        A first shape, ``(n_points, 2)``, for an image that has no landmarks.

        ``box`` is ``[[row_min, col_min], [row_max, col_max]]`` if the annotator drew where the
        object is (the shape should fill it); the model chooses a box itself otherwise.
        """
        raise NotImplementedError

    def suggest_box(self, state: Any, image: Any) -> np.ndarray:
        r"""
        Where the model would look for the object, ``[[row_min, col_min], [row_max, col_max]]``
        (what the client shows as the suggested box). By default the box of what
        :meth:`initialize` gives.
        """
        shape = np.asarray(self.initialize(state, image))
        return np.stack([shape.min(axis=0), shape.max(axis=0)])

    def prepare(self, image: Any, training: bool) -> Any:
        r"""
        Condition a freshly imported image. The default does nothing; ``training`` is ``True``
        for images the model learns from, which can be cropped and shrunk to save time.
        """
        return image

    def load(self, path: Path) -> Any:
        r"""Read a state that was saved with :meth:`save` (menpo's pickle by default)."""
        import menpo.io as mio

        return mio.import_pickle(path)

    def save(self, state: Any, path: Path) -> None:
        r"""Write a state so that :meth:`load` reads it."""
        import menpo.io as mio

        mio.export_pickle(state, path, overwrite=True)

    # what this model can do, from what it overrides
    def _has(self, name: str) -> bool:
        return getattr(type(self), name) is not getattr(FitModel, name)

    @property
    def can_build(self) -> bool:
        return self._has("build")

    @property
    def can_increment(self) -> bool:
        return self._has("increment")

    @property
    def can_initialize(self) -> bool:
        return self._has("initialize")


class AutoFit:
    r"""
    The model that learns from saved landmarks and fits new images.

    Parameters
    ----------
    ids_to_paths : `dict` of `str` to `Path`
        The image of every asset id. The landmarks are the ``.ljson`` next to it.
    model : :class:`FitModel`
        What to learn with.
    initial_state : optional
        A state that the model starts from (see :meth:`FitModel.load`). With one the model can
        fit at once; without it there is no model until it has been trained.
    min_images : `int`, optional
        Reviewed images needed to build a model when there is no initial state.
    from_saved : `bool`, optional
        If ``True`` only images that were saved through the server (``"reviewed": true``)
        are learnt from; if ``False`` every image that has complete landmarks is.
    learn_on_save : `bool`, optional
        Learn from each image as it is saved (needs a model that can ``increment``). Otherwise
        the model only learns when :meth:`train` is called.
    save_path : `Path`, optional
        Where to write the state after each update, so that learning survives a restart
        (use it as ``initial_state`` next time).
    background : `bool`, optional
        Run in a background thread (``False`` is for tests: everything finishes before returning).
    """

    def __init__(
        self,
        ids_to_paths: Dict[str, Path],
        model: FitModel,
        initial_state: Any = None,
        min_images: Optional[int] = None,
        from_saved: bool = True,
        learn_on_save: bool = True,
        save_path: Optional[Path] = None,
        background: bool = True,
    ) -> None:
        self.ids_to_paths = ids_to_paths
        self.model = model
        self.min_images = model.min_images if min_images is None else min_images
        self.from_saved = from_saved
        self.learn_on_save = learn_on_save and model.can_increment
        self.save_path = save_path
        self._initial = initial_state
        self._lock = threading.Lock()  # the state and the fitter
        self._work = threading.Lock()  # one learning step at a time
        self._executor = ThreadPoolExecutor(max_workers=1) if background else None
        self._pending = False
        self._job: Optional[Dict[str, Any]] = None  # the latest job: train or predict
        self._state: Any = initial_state
        self._fitter: Any = None
        self._learnt: Dict[
            str, float
        ] = {}  # asset id -> modification time of what was learnt
        self._error: Optional[str] = None
        if initial_state is not None:
            self._fitter = model.fitter(initial_state)  # fits before anything is learnt

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

    def _may_learn_from(self, asset_id: str) -> bool:
        """Complete landmarks, and saved through the server if ``from_saved``."""
        data = self._read(asset_id)
        if self._complete_points(data) is None:
            return False
        return not self.from_saved or bool((data or {}).get("reviewed"))

    def annotated_ids(self) -> List[str]:
        """The assets that the model learns from."""
        return [i for i in sorted(self.ids_to_paths) if self._may_learn_from(i)]

    def unlabelled_ids(self) -> List[str]:
        """The assets with no complete landmarks."""
        return [
            i
            for i in sorted(self.ids_to_paths)
            if self._complete_points(self._read(i)) is None
        ]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            ready = self._fitter is not None
            learnt = len(self._learnt)
        needed = 0 if self._initial is not None else self.min_images
        return {
            "ready": ready,
            "model": type(self.model).__name__,
            "can_increment": self.model.can_increment,
            "can_build": self.model.can_build,
            "can_initialize": self.model.can_initialize,
            "learn_on_save": self.learn_on_save,
            "has_initial_state": self._initial is not None,
            "trained_on": learnt,
            "annotated": len(self.annotated_ids()),
            "unlabelled": len(self.unlabelled_ids()),
            "min_images": needed,
            "from_saved": self.from_saved,
            "job": self._job,
            "error": self._error,
        }

    # ---- learning
    def notify_saved(self, asset_id: str) -> None:
        """Landmarks were saved: learn from them (in the background), if that is switched on."""
        if not self.learn_on_save:
            return
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

    def _set_state(self, state: Any, learnt: Dict[str, float]) -> None:
        fitter = self.model.fitter(state)
        with self._lock:
            self._state, self._fitter, self._learnt = state, fitter, learnt
            self._error = None
        if self.save_path is not None:
            self.model.save(state, self.save_path)

    def refresh(self) -> None:
        """Learn from whatever is new: add new images, or start over after a change."""
        with self._work:
            ids = self.annotated_ids()
            needed = 0 if self._initial is not None else self.min_images
            if len(ids) < max(needed, 1):
                return
            mtimes = {i: self._ljson_path(i).stat().st_mtime for i in ids}
            with self._lock:
                learnt, state = dict(self._learnt), self._state
            new = [i for i in ids if i not in learnt]
            changed = [i for i in learnt if i in mtimes and mtimes[i] > learnt[i]]
            gone = [i for i in learnt if i not in mtimes]
            if learnt and not (new or changed or gone):
                return
            if not learnt or changed or gone:
                state = self._learn_from_start(ids)
                learnt = {}
            else:
                logger.info("adding {} images to the model", len(new))
                state = self.model.increment(state, self._load(new))
            learnt.update({i: mtimes[i] for i in ids})
            self._set_state(state, learnt)

    def _learn_from_start(self, ids: Sequence[str], progress: Any = None) -> Any:
        # an incremental model cannot forget: start from the initial state, or from nothing
        if self._initial is not None and self.model.can_increment:
            logger.info("adding {} images to the initial state", len(ids))
            return self.model.increment(self._initial, self._load(ids, progress))
        if self.model.can_build:
            logger.info("building the model from {} images", len(ids))
            return self.model.build(self._load(ids, progress))
        raise NotImplementedError(
            f"{type(self.model).__name__} can neither build nor increment: it cannot learn"
        )

    # ---- jobs the annotator asks for
    def _start_job(self, kind: str, run: Any, total: int) -> Dict[str, Any]:
        if self._job and self._job["state"] == "running":
            raise Busy(f"a {self._job['kind']} job is running")
        job: Dict[str, Any] = {
            "kind": kind,
            "state": "running",
            "done": 0,
            "total": total,
        }
        self._job = job

        def tick(n: int = 1) -> None:
            job["done"] += n

        def work() -> None:
            try:
                job["message"] = run(tick)
                job["state"] = "done"
            except Exception as error:  # noqa: BLE001
                job["state"] = "error"
                job["message"] = f"{type(error).__name__}: {error}"
                logger.exception("{} failed", kind)

        if self._executor is None:
            work()
        else:
            self._executor.submit(work)
        return job

    def set_learn_on_save(self, enabled: bool) -> None:
        r"""Switch learning from each saved image on or off (only a model that can ``increment``)."""
        self.learn_on_save = bool(enabled) and self.model.can_increment

    def learn(self, asset_id: str) -> Dict[str, Any]:
        r"""
        Add one image to the model, as a job. Only a **saved** image can be learnt from: the
        landmarks that are in its file, not the ones on the annotator's screen.

        If the image was learnt from before and has been saved again since, the model starts
        over from its initial state with the images it had learnt (an incremental model cannot
        forget the old landmarks).

        Raises
        ------
        NothingToLearn
            The model cannot ``increment``, the image is not saved (as complete, reviewed
            landmarks), or there is no state to add it to yet (use :meth:`train`).
        """
        if not self.model.can_increment:
            raise NothingToLearn(
                f"{type(self.model).__name__} cannot learn one image at a time "
                "(it has no increment())"
            )
        if not self._may_learn_from(asset_id):
            raise NothingToLearn(
                "the model learns only from saved landmarks: place every landmark and "
                f"save {asset_id} first"
                if self.from_saved
                else f"{asset_id} has no complete landmarks to learn from"
            )
        if self._fitter is None:
            raise NothingToLearn(self._not_ready_message())

        def run(tick: Any) -> str:
            with self._work:
                mtime = self._ljson_path(asset_id).stat().st_mtime
                with self._lock:
                    learnt, state = dict(self._learnt), self._state
                if learnt.get(asset_id, -1.0) >= mtime:
                    tick()
                    return f"already learnt from {asset_id}"
                if asset_id in learnt:
                    # saved again: the old landmarks are in the state, start over without them
                    ids = [i for i in learnt if self._ljson_path(i).exists()]
                    ids = sorted(set(ids) | {asset_id})
                    state = self._learn_from_start(ids)
                    learnt = {i: self._ljson_path(i).stat().st_mtime for i in ids}
                else:
                    state = self.model.increment(state, self._load([asset_id]))
                    learnt[asset_id] = mtime
                self._set_state(state, learnt)
                tick()
            return f"learnt from {asset_id}"

        return self._start_job("learn", run, 1)

    def train(self) -> Dict[str, Any]:
        r"""
        Learn from all the annotations again, as a job: from the initial state if there is one,
        else from scratch. Works for a model that can ``build`` (or ``increment`` an initial
        state), and is the way to start from no annotations at all.
        """
        ids = self.annotated_ids()
        if not ids:
            raise NothingToLearn("there are no annotations to learn from")
        if not (
            self.model.can_build
            or (self._initial is not None and self.model.can_increment)
        ):
            raise NothingToLearn(f"{type(self.model).__name__} cannot be trained")

        def run(tick: Any) -> str:
            with self._work:
                state = self._learn_from_start(ids, tick)
                self._set_state(
                    state, {i: self._ljson_path(i).stat().st_mtime for i in ids}
                )
            return f"learnt from {len(ids)} images"

        return self._start_job("train", run, len(ids))

    def predict(self, skeleton: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Fit every image that has no landmarks and write its ``.ljson`` (``"reviewed": false``,
        so it does not count as an annotation until saved through the server), as a job.

        Parameters
        ----------
        skeleton : `dict`
            The LJSON of the template: its labels and connectivity are copied to each file.
        """
        if self._fitter is None:
            raise NotReady(self._not_ready_message())
        if not self.model.can_initialize:
            raise NothingToLearn(
                f"{type(self.model).__name__} cannot start from nothing (no initialize())"
            )
        ids = self.unlabelled_ids()

        def run(tick: Any) -> str:
            import menpo.io as mio

            done = 0
            for asset_id in ids:
                with self._lock:
                    state, fitter = self._state, self._fitter
                image = self.model.prepare(
                    mio.import_image(self.ids_to_paths[asset_id]), False
                )
                start = np.asarray(self.model.initialize(state, image))
                points = np.asarray(self.model.fit(fitter, image, start))
                data = copy.deepcopy(skeleton)
                data["landmarks"]["points"] = points.tolist()
                data["reviewed"] = False
                self._ljson_path(asset_id).write_text(
                    json.dumps(data, indent=4, sort_keys=True)
                )
                done += 1
                tick()
            return f"fitted {done} images"

        return self._start_job("predict", run, len(ids))

    # ---- fitting
    def _not_ready_message(self) -> str:
        if self.model.can_build or self.model.can_increment:
            return (
                f"the model needs {self.min_images} reviewed images "
                f"(save the landmarks of a few more, or train it); "
                f"it has learnt from {len(self._learnt)}"
            )
        return "the model has no state: give it an initial state in the fit config"

    def fit(
        self,
        asset_id: str,
        start: Optional[Sequence[Sequence[Optional[float]]]] = None,
        box: Optional[Sequence[Sequence[float]]] = None,
    ) -> np.ndarray:
        r"""
        Refine the landmarks of an image. Learns nothing and saves nothing.

        Parameters
        ----------
        asset_id : `str`
            The image.
        start : ``(n_points, 2)`` array-like, optional
            Where the landmarks are now (what is on the annotator's screen, saved or not).
            The ``.ljson`` of the image if not given.
        box : ``[[row_min, col_min], [row_max, col_max]]``, optional
            A box the annotator drew around the object. The fit starts from the model's initial
            shape in it, whatever landmarks there are (a poor start is not worth refining).
            Without a box the fit starts from ``start`` (the current annotation) if all of its
            points are placed, and from the model's own initial shape if not.

        Returns
        -------
        points : ``(n_points, 2)`` `ndarray`
            The new positions, (row, column) like the points of the LJSON.

        Raises
        ------
        NotReady
            No model yet.
        NothingToRefine
            There are no complete landmarks to start from, and the model cannot
            ``initialize`` one.
        """
        with self._lock:
            state, fitter = self._state, self._fitter
        if fitter is None:
            raise NotReady(self._not_ready_message())
        if start is None:
            points = self._complete_points(self._read(asset_id))
        else:
            points = self._complete_points({"landmarks": {"points": start}})
        import menpo.io as mio

        image = self.model.prepare(mio.import_image(self.ids_to_paths[asset_id]), False)
        if box is not None:
            if not self.model.can_initialize:
                raise NothingToRefine(
                    f"{type(self.model).__name__} cannot start from a box"
                )
            points = np.asarray(self.model.initialize(state, image, _box(box)))
        elif points is None:
            if not self.model.can_initialize:
                raise NothingToRefine(
                    "the image has no landmarks to refine: place them, or copy the previous ones"
                )
            points = np.asarray(self.model.initialize(state, image))
        return np.asarray(self.model.fit(fitter, image, points))

    def suggest_box(self, asset_id: str) -> np.ndarray:
        r"""The box the model would start from for an image, to draw and let the annotator change."""
        with self._lock:
            state = self._state
        if state is None:
            raise NotReady(self._not_ready_message())
        if not self.model.can_initialize:
            raise NothingToRefine(
                f"{type(self.model).__name__} cannot start from a box"
            )
        import menpo.io as mio

        image = self.model.prepare(mio.import_image(self.ids_to_paths[asset_id]), False)
        return np.asarray(self.model.suggest_box(state, image))

    def _load(self, asset_ids: Sequence[str], progress: Any = None) -> List[Any]:
        import menpo.io as mio
        from menpo.shape import PointCloud

        images = []
        for asset_id in asset_ids:
            image = mio.import_image(self.ids_to_paths[asset_id])
            # the landmarks are the reviewed ones, whatever else the folder has
            points = self._complete_points(self._read(asset_id))
            image.landmarks[LJSON_GROUP] = PointCloud(points)
            images.append(self.model.prepare(image, True))
            if progress is not None:
                progress()
        return images


CONFIG_KEYS = {
    "model",
    "options",
    "initial_state",
    "save_state",
    "from_saved",
    "learn_on_save",
    "min_images",
}


def autofit_from_config(
    ids_to_paths: Dict[str, Path], config_path: Optional[Path] = None
) -> AutoFit:
    r"""
    Make the :class:`AutoFit` that a config file describes (the defaults without one).

    .. code-block:: yaml

        model: patch-aam             # a short name (patch-aam, holistic-aam) or package.module:Class
        options:                     # given to the model's constructor
          patch_sizes: [15, 23]
          diagonal: 150
        initial_state: faces.pkl     # optional: start from this (read by the model's load())
        save_state: learnt.pkl       # optional: written after each update
        from_saved: true             # false: learn from every image with complete landmarks
        learn_on_save: true          # learn from each image as it is saved (needs increment)
        min_images: 8                # reviewed images to build a model without an initial state

    Paths are relative to the config file.
    """
    import yaml

    config: Dict[str, Any] = {}
    base = Path.cwd()
    if config_path is not None:
        base = Path(config_path).resolve().parent
        config = yaml.safe_load(Path(config_path).read_text()) or {}
        if not isinstance(config, dict):
            raise ValueError(f"{config_path} must be a mapping of settings")
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"unknown settings {sorted(unknown)} in {config_path}; "
                f"known: {sorted(CONFIG_KEYS)}"
            )
    model = load_model_class(config.get("model", "patch-aam"))(
        **(config.get("options") or {})
    )
    initial = None
    if config.get("initial_state"):
        initial = model.load(base / config["initial_state"])
    save_path = base / config["save_state"] if config.get("save_state") else None
    return AutoFit(
        ids_to_paths,
        model,
        initial_state=initial,
        min_images=config.get("min_images"),
        from_saved=bool(config.get("from_saved", True)),
        learn_on_save=bool(config.get("learn_on_save", True)),
        save_path=save_path,
    )


def _box(box: Sequence[Sequence[float]]) -> np.ndarray:
    r"""A box as ``[[row_min, col_min], [row_max, col_max]]`` from any two opposite corners."""
    corners = np.asarray(box, dtype=float)
    if corners.shape != (2, 2) or not np.isfinite(corners).all():
        raise NothingToRefine("a box is two corners, each [row, column]")
    ordered = np.stack([corners.min(axis=0), corners.max(axis=0)])
    if (ordered[1] - ordered[0] <= 1).any():
        raise NothingToRefine("the box is too small")
    return ordered


def load_model_class(spec: str) -> type:
    r"""
    The model class for the ``model:`` setting of the config: ``package.module:Class``, or one
    of the short names in :data:`landmarkerio.fit_models.MODELS`, each of which stands for a
    class of that module (``patch-aam`` is ``landmarkerio.fit_models:PatchAAM``).
    """
    from importlib import import_module

    from landmarkerio import fit_models

    if spec in fit_models.MODELS:
        return fit_models.MODELS[spec]
    if ":" not in spec:
        raise ValueError(
            f"'{spec}' is not one of {sorted(fit_models.MODELS)}; "
            "give a class as package.module:Class"
        )
    module, _, name = spec.partition(":")
    cls = getattr(import_module(module), name)
    if not (isinstance(cls, type) and issubclass(cls, FitModel)):
        raise TypeError(f"{spec} is not a subclass of landmarkerio.fit.FitModel")
    return cls


def require_fit_dependencies() -> None:
    """Raise a helpful error if menpofit is not installed."""
    try:
        import menpofit  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Auto-fit needs menpofit: pip install 'landmarkerio-server[fit]'"
        ) from error


__all__ = [
    "AutoFit",
    "Busy",
    "FitModel",
    "NothingToLearn",
    "NotReady",
    "NothingToRefine",
    "autofit_from_config",
    "load_model_class",
    "require_fit_dependencies",
]
