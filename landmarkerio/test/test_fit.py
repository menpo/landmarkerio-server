import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest
from PIL import Image
from sanic import Sanic

from landmarkerio.cache import cache_assets, filepath_as_asset_id_under_dir
from landmarkerio.fit import (
    AutoFit,
    Busy,
    FitModel,
    FitRegistry,
    NothingToLearn,
    NothingToRefine,
    NotReady,
    Proposals,
    UnknownModel,
    autofit_from_config,
    registry_from_config,
)
from landmarkerio.landmark import InplaceFileLmAdapter, ReviewedLmAdapter
from landmarkerio.servers.serve import serve_from_cache


class Minimal(FitModel):
    """The least a model must do: fit. It was trained elsewhere."""

    def __init__(self):
        self.min_images = 3

    def fitter(self, state):
        return state

    def fit(self, fitter, image, start):
        return start + fitter  # moved by the state, a number


class Full(Minimal):
    """A model that is a list of the assets it saw, and everything optional."""

    def __init__(self, min_images=3):
        self.min_images = min_images
        self.log: List[Any] = []

    def fitter(self, state):
        return len(state)  # moved by the number of images the state saw

    def build(self, images):
        self.log.append(("build", sorted(images)))
        return list(images)

    def increment(self, state, images):
        self.log.append(("add", sorted(images)))
        return list(state) + list(images)

    def initialize(self, state, image, box=None):
        if box is None:
            return np.zeros((1, 2))
        return box.mean(axis=0, keepdims=True)  # the middle of the box


class Detector(Minimal):
    """A pretrained fitter that also finds where to start (nothing to learn)."""

    def initialize(self, state, image, box=None):
        return np.zeros((1, 2))


class Incrementing(Full):
    """Cannot build from nothing, only grow a state it was given."""

    build = FitModel.build


class FakeAutoFit(AutoFit):
    """Learns from the names of the assets, which is what the Full model counts."""

    def _load(self, asset_ids, progress=None):
        for _ in asset_ids:
            if progress:
                progress()
        return list(asset_ids)


def write(folder: Path, name: str, points, reviewed: Optional[bool] = True) -> None:
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(folder / f"{name}.png")
    if points is None:
        return
    data = {
        "version": 2,
        "labels": [{"label": "all", "mask": list(range(len(points)))}],
        "landmarks": {"points": points, "connectivity": []},
    }
    if reviewed is not None:
        data["reviewed"] = reviewed
    (folder / f"{name}.ljson").write_text(json.dumps(data))


def make(folder: Path, model=None, **kwargs) -> FakeAutoFit:
    ids = {p.name: p for p in sorted(folder.glob("*.png"))}
    kwargs.setdefault("background", False)
    return FakeAutoFit(ids, model or Full(), **kwargs)


def reviewed(folder: Path, n: int, points=((1.0, 2.0),)) -> None:
    for i in range(n):
        write(folder, f"a{i}", [list(p) for p in points])


# ---- learning
@pytest.fixture
def folder(tmp_path):
    return tmp_path


def test_no_model_until_enough_images_are_reviewed(folder):
    reviewed(folder, 2)
    fit = make(folder, min_images=3)
    fit.refresh()
    assert fit.status()["ready"] is False and fit.status()["annotated"] == 2
    with pytest.raises(NotReady, match="needs 3 reviewed images"):
        fit.fit("a0.png")


def test_only_reviewed_images_count_unless_asked(folder):
    write(folder, "r", [[1.0, 2.0]])
    write(folder, "u", [[1.0, 2.0]], reviewed=None)
    write(folder, "partial", [[1.0, None]])
    assert make(folder).annotated_ids() == ["r.png"]
    assert make(folder, from_saved=False).annotated_ids() == ["r.png", "u.png"]


def test_builds_then_adds_new_images_to_the_model(folder):
    reviewed(folder, 3)
    fit = make(folder, min_images=3)
    fit.refresh()
    assert fit.model.log == [("build", ["a0.png", "a1.png", "a2.png"])]
    fit.refresh()  # nothing new: nothing done
    assert len(fit.model.log) == 1
    write(folder, "a3", [[1.0, 2.0]])
    fit.ids_to_paths["a3.png"] = folder / "a3.png"
    fit.refresh()
    assert fit.model.log[-1] == ("add", ["a3.png"])  # incremental, not from scratch
    assert fit.status()["trained_on"] == 4


def test_a_resaved_image_that_was_learnt_from_starts_over(folder):
    reviewed(folder, 3)
    fit = make(folder, min_images=3)
    fit.refresh()
    path = folder / "a1.ljson"
    later = os.stat(path).st_mtime + 10
    write(folder, "a1", [[9.0, 9.0]])
    os.utime(path, (later, later))
    fit.refresh()
    assert fit.model.log[-1] == ("build", ["a0.png", "a1.png", "a2.png"])


# ---- an initial state: fit before anything is labelled
def test_an_initial_state_fits_at_once_and_learns_on_top_of_it(folder):
    write(folder, "new", [[10.0, 20.0]], reviewed=None)
    fit = make(folder, initial_state=["x", "y"], from_saved=True)
    status = fit.status()
    assert status["ready"] and status["has_initial_state"] and status["min_images"] == 0
    assert status["trained_on"] == 0
    assert fit.fit("new.png").tolist() == [[12.0, 22.0]]  # fitted by the 2-image state
    # a single reviewed image is enough to grow it (no minimum with an initial state)
    write(folder, "one", [[1.0, 2.0]])
    fit.ids_to_paths["one.png"] = folder / "one.png"
    fit.refresh()
    assert fit.model.log == [("add", ["one.png"])]
    assert fit.fit("new.png").tolist() == [[13.0, 23.0]]


def test_starting_over_starts_from_the_initial_state(folder):
    reviewed(folder, 2)
    fit = make(folder, initial_state=["x"])
    fit.refresh()
    path = folder / "a0.ljson"
    later = os.stat(path).st_mtime + 10
    write(folder, "a0", [[9.0, 9.0]])
    os.utime(path, (later, later))
    fit.refresh()
    # not from scratch (that would lose the initial state): initial + both images
    assert fit.model.log[-1] == ("add", ["a0.png", "a1.png"])
    assert fit.status()["trained_on"] == 2


def test_a_fit_only_model_fits_and_cannot_learn(folder):
    reviewed(folder, 4)
    write(folder, "new", [[10.0, 20.0]], reviewed=None)
    fit = make(folder, Minimal(), initial_state=5)
    status = fit.status()
    assert (status["can_increment"], status["can_build"], status["can_initialize"]) == (
        False,
        False,
        False,
    )
    assert status["learn_on_save"] is False
    fit.notify_saved("a0.png")  # nothing to learn with: nothing happens
    assert fit.status()["trained_on"] == 0
    assert fit.fit("new.png").tolist() == [[15.0, 25.0]]
    with pytest.raises(NothingToLearn, match="cannot be trained"):
        fit.train()
    # and it has no way to fit an image that has no landmarks
    write(folder, "bare", None)
    fit.ids_to_paths["bare.png"] = folder / "bare.png"
    with pytest.raises(NothingToRefine, match="no landmarks to refine"):
        fit.fit("bare.png")


def test_a_model_without_a_state_says_so(folder):
    write(folder, "new", [[1.0, 2.0]], reviewed=None)
    with pytest.raises(NotReady, match="initial state"):
        make(folder, Minimal()).fit("new.png")


# ---- where a fit starts
def test_a_fit_starts_from_the_current_landmarks_a_box_or_the_models_own_guess(folder):
    write(folder, "saved", [[100.0, 100.0]], reviewed=None)
    write(folder, "bare", None)
    fit = make(folder, initial_state=[])  # moves by 0: the output is the start
    fit.ids_to_paths["bare.png"] = folder / "bare.png"
    # 1. the current landmarks, whatever the file says
    assert fit.fit("saved.png", [[3.0, 4.0]]).tolist() == [[3.0, 4.0]]
    # ... or the file's, if the client sent none
    assert fit.fit("saved.png").tolist() == [[100.0, 100.0]]
    # 2. nothing placed: the model decides
    assert fit.fit("bare.png").tolist() == [[0.0, 0.0]]
    assert fit.fit("saved.png", [[None, None]]).tolist() == [[0.0, 0.0]]
    # 3. a box wins over landmarks that are there (a poor start is not worth refining)
    box = [[10.0, 20.0], [30.0, 60.0]]
    assert fit.fit("saved.png", [[3.0, 4.0]], box).tolist() == [[20.0, 40.0]]
    # any two opposite corners do
    assert fit.fit("bare.png", None, [[30.0, 60.0], [10.0, 20.0]]).tolist() == [
        [20.0, 40.0]
    ]
    with pytest.raises(NothingToRefine, match="too small"):
        fit.fit("bare.png", None, [[10.0, 20.0], [10.5, 60.0]])
    with pytest.raises(NothingToRefine, match="two corners"):
        fit.fit("bare.png", None, [[10.0, 20.0]])


# ---- jobs
def test_train_learns_from_all_the_annotations_as_a_job(folder):
    reviewed(folder, 4)
    fit = make(folder, min_images=100, learn_on_save=False)  # never learns on its own
    fit.refresh()
    assert fit.status()["ready"] is False
    job = fit.train()
    assert job["state"] == "done" and job["done"] == job["total"] == 4
    assert job["message"] == "learnt from 4 images"
    assert fit.model.log == [("build", [f"a{i}.png" for i in range(4)])]
    assert fit.status()["ready"] and fit.status()["trained_on"] == 4


def test_learn_adds_one_saved_image_and_only_that_one(folder):
    reviewed(folder, 3)
    fit = make(folder, min_images=3, learn_on_save=False)
    fit.train()
    write(folder, "extra", [[1.0, 2.0]])
    fit.ids_to_paths["extra.png"] = folder / "extra.png"
    write(folder, "other", [[1.0, 2.0]])
    fit.ids_to_paths["other.png"] = folder / "other.png"
    job = fit.learn("extra.png")
    assert job["state"] == "done" and job["message"] == "learnt from extra.png"
    assert fit.model.log[-1] == ("add", ["extra.png"])  # not other.png
    assert fit.status()["trained_on"] == 4
    assert fit.learn("extra.png")["message"] == "already learnt from extra.png"
    assert len(fit.model.log) == 2  # and nothing was done for it


def test_learn_only_from_a_saved_image(folder):
    reviewed(folder, 3)
    write(
        folder, "unsaved", [[1.0, 2.0]], reviewed=None
    )  # has landmarks, not saved here
    write(folder, "partial", [[1.0, None]])
    write(folder, "none", None)
    fit = make(folder, min_images=3, learn_on_save=False)
    fit.train()
    for name in ("unsaved", "partial", "none"):
        fit.ids_to_paths[f"{name}.png"] = folder / f"{name}.png"
        with pytest.raises(NothingToLearn, match="learns only from saved landmarks"):
            fit.learn(f"{name}.png")
    assert fit.status()["trained_on"] == 3
    # unless it was asked to learn from every file
    fit.from_saved = False
    assert fit.learn("unsaved.png")["state"] == "done"
    with pytest.raises(NothingToLearn, match="no complete landmarks"):
        fit.learn("partial.png")


def test_learn_a_resaved_image_starts_over_without_its_old_landmarks(folder):
    reviewed(folder, 3)
    fit = make(folder, min_images=3, learn_on_save=False)
    fit.train()
    time.sleep(0.01)
    write(folder, "a1", [[5.0, 6.0]])  # saved again
    assert fit.learn("a1.png")["state"] == "done"
    assert fit.model.log[-1] == ("build", ["a0.png", "a1.png", "a2.png"])
    assert fit.status()["trained_on"] == 3


def test_learn_needs_a_model_that_increments_and_a_state_to_add_to(folder):
    reviewed(folder, 3)
    fit = make(folder, Minimal(), learn_on_save=False)
    with pytest.raises(NothingToLearn, match="cannot learn one image at a time"):
        fit.learn("a0.png")
    fit = make(folder, min_images=100, learn_on_save=False)  # no state yet
    with pytest.raises(NothingToLearn, match="needs 100 reviewed images"):
        fit.learn("a0.png")


def test_learn_on_save_can_be_switched(folder):
    fit = make(folder)
    assert fit.learn_on_save
    fit.set_learn_on_save(False)
    assert fit.status()["learn_on_save"] is False
    fit.set_learn_on_save(True)
    assert fit.status()["learn_on_save"] is True
    only = make(folder, Minimal())
    only.set_learn_on_save(True)
    assert only.learn_on_save is False  # nothing to learn with


def test_short_names_are_the_classes_of_fit_models():
    from landmarkerio import fit_models
    from landmarkerio.fit import load_model_class

    for name, cls in fit_models.MODELS.items():
        assert load_model_class(name) is cls
        assert load_model_class(f"landmarkerio.fit_models:{cls.__name__}") is cls
    with pytest.raises(ValueError, match="patch-aam"):
        load_model_class("nonsense")


def test_train_without_annotations_is_refused(tmp_path):
    write(tmp_path, "a", [[1.0, 2.0]], reviewed=None)
    with pytest.raises(NothingToLearn, match="no annotations"):
        make(tmp_path).train()


def test_an_incrementing_model_trains_from_its_initial_state_only(folder):
    reviewed(folder, 2)
    fit = make(folder, Incrementing(), initial_state=["x"], learn_on_save=False)
    assert fit.train()["state"] == "done"
    assert fit.model.log == [("add", ["a0.png", "a1.png"])]
    with pytest.raises(NothingToLearn, match="cannot be trained"):
        make(folder, Incrementing()).train()  # no initial state and cannot build


def test_a_failed_job_is_reported_not_raised(folder):
    reviewed(folder, 3)

    class Broken(Full):
        def build(self, images):
            raise RuntimeError("boom")

    fit = make(folder, Broken(), learn_on_save=False)
    job = fit.train()
    assert job["state"] == "error" and "boom" in job["message"]
    assert fit.status()["ready"] is False and fit.status()["job"] == job


def test_a_second_job_while_one_runs_is_refused(folder):
    reviewed(folder, 3)
    fit = make(folder)
    fit._job = {"kind": "train", "state": "running", "done": 1, "total": 3}
    with pytest.raises(Busy, match="train job is running"):
        fit.train()


SKELETON = {
    "version": 2,
    "labels": [{"label": "all", "mask": [0]}],
    "landmarks": {"points": [[None, None]], "connectivity": []},
}


def test_predict_is_a_prefit_that_keeps_proposals_apart_from_saved_landmarks(folder):
    reviewed(folder, 3)
    write(folder, "bare1", None)
    write(folder, "bare2", None)
    fit = make(folder)
    fit.refresh()  # built from 3
    job = fit.predict(SKELETON)
    assert (
        job["state"] == "done"
        and job["total"] == 2
        and job["message"] == "fitted 2 images"
    )
    for name in ("bare1", "bare2"):
        assert not (folder / f"{name}.ljson").exists()  # nothing saved
        data = json.loads((folder / ".prefit" / "Full" / f"{name}.ljson").read_text())
        assert data["landmarks"]["points"] == [
            [3.0, 3.0]
        ]  # initialize (0) + a 3-image state
        assert data["reviewed"] is False and data["labels"] == SKELETON["labels"]
        assert data["source"] == "prefit:Full"
    # a proposal is not an annotation and the image is still unlabelled
    assert fit.annotated_ids() == ["a0.png", "a1.png", "a2.png"]
    assert fit.unlabelled_ids() == ["bare1.png", "bare2.png"]
    assert SKELETON["landmarks"]["points"] == [[None, None]]
    assert fit.status()["proposals"] == 2


def test_prefit_skips_proposals_unless_overwriting_and_can_chain_frames(folder):
    for i, name in enumerate(["f1", "f2", "f3"]):
        write(folder, name, None)
    write(folder, "f0", [[10.0, 10.0]], reviewed=True)
    reviewed(folder, 2)
    fit = make(folder, min_images=2)
    fit.refresh()  # 3 images: a0, a1, f0 -> fits move by 3
    fit.predict(SKELETON)
    path = lambda n: folder / ".prefit" / "Full" / f"{n}.ljson"  # noqa: E731
    first = json.loads(path("f1").read_text())["landmarks"]["points"]
    assert first == [[3.0, 3.0]]  # from the model's own start
    path("f1").write_text(path("f1").read_text().replace("3.0", "9.0"))
    assert fit.predict(SKELETON)["total"] == 0  # all have proposals
    assert json.loads(path("f1").read_text())["landmarks"]["points"] == [[9.0, 9.0]]
    # chained: f1 starts from the saved f0 (frames in order), f2 from f1, ...
    fit.predict(SKELETON, overwrite=True, chain=True)
    got = [
        json.loads(path(n).read_text())["landmarks"]["points"][0][0]
        for n in ("f1", "f2", "f3")
    ]
    assert got == [13.0, 16.0, 19.0]


def test_a_proposal_is_served_until_saved_and_saving_it_as_it_is_accepts_it(folder):
    write(folder, "a", None)
    ids = {"a": folder / "a.png"}
    proposals = Proposals(ids)
    told = []
    adapter = ReviewedLmAdapter(InplaceFileLmAdapter(ids), told.append, proposals)
    with pytest.raises(FileNotFoundError):
        adapter.load_landmark("a", "inplace")
    assert adapter.asset_id_to_lm_id() == {}
    proposals.write("m1", "a", {**BODY, "reviewed": False, "source": "prefit:m1"})
    assert adapter.asset_id_to_lm_id() == {"a": ["inplace"]}
    served = adapter.load_landmark("a", "inplace")
    assert served["reviewed"] is False and served["source"] == "prefit:m1"
    # nothing is saved, so nothing is learnt from
    assert not (folder / "a.ljson").exists() and told == []
    # accept it as it is
    adapter.save_landmark("a", "inplace", {k: v for k, v in served.items()})
    data = json.loads((folder / "a.ljson").read_text())
    assert data["reviewed"] is True and data["fitted_by"] == "m1"
    assert "source" not in data and told == ["a"]
    assert not proposals.has("a")  # the proposal is gone
    # saved again later: still says who started it, and the saved one is what is served
    adapter.save_landmark("a", "inplace", BODY)
    assert json.loads((folder / "a.ljson").read_text())["fitted_by"] == "m1"
    assert adapter.load_landmark("a", "inplace")["reviewed"] is True


def test_the_newest_proposal_is_the_one_served(folder):
    write(folder, "a", None)
    proposals = Proposals({"a": folder / "a.png"})
    proposals.write("old", "a", {"v": 1})
    os.utime(proposals.path("old", "a"), (1, 1))
    proposals.write("new", "a", {"v": 2})
    assert proposals.models("a") == ["new", "old"]
    assert proposals.read("a") == {"v": 2} and proposals.read("a", "old") == {"v": 1}
    assert proposals.take("a") == "new" and proposals.ids() == []


def test_predict_needs_a_model_that_can_start_from_nothing(folder):
    reviewed(folder, 1)
    skeleton = {
        "version": 2,
        "labels": [],
        "landmarks": {"points": [], "connectivity": []},
    }
    with pytest.raises(NothingToLearn, match="cannot start from nothing"):
        make(folder, Minimal(), initial_state=0).predict(skeleton)
    with pytest.raises(NotReady):
        make(folder).predict(skeleton)


# ---- background and hooks
def test_background_learning_finishes_and_a_failure_is_reported(folder):
    reviewed(folder, 3)
    ids = {p.name: p for p in folder.glob("*.png")}
    fit = FakeAutoFit(ids, Full(), min_images=3, background=True)
    fit.notify_saved("a0.png")
    fit._executor.shutdown(wait=True)
    assert fit.status()["ready"] is True

    class Broken(Full):
        def build(self, images):
            raise RuntimeError("boom")

    broken = FakeAutoFit(ids, Broken(), min_images=3, background=True)
    broken.notify_saved("a0.png")
    broken._executor.shutdown(wait=True)
    assert "boom" in broken.status()["error"]


def test_saving_marks_reviewed_and_tells_the_fitter(folder):
    write(folder, "a", [[1.0, 2.0]], reviewed=None)
    told = []
    adapter = ReviewedLmAdapter(
        InplaceFileLmAdapter({"a": folder / "a.png"}), told.append
    )
    adapter.save_landmark(
        "a", "inplace", {"version": 2, "landmarks": {"points": [[5.0, 6.0]]}}
    )
    assert json.loads((folder / "a.ljson").read_text())["reviewed"] is True
    assert told == ["a"]


def test_the_state_is_saved_after_learning(folder):
    reviewed(folder, 3)
    saved = []

    class Saving(Full):
        def save(self, state, path):
            saved.append((list(state), path))

    fit = make(folder, Saving(), save_path=folder / "state.pkl")
    fit.refresh()
    assert saved == [(["a0.png", "a1.png", "a2.png"], folder / "state.pkl")]


# ---- the config file
class FromConfig(Minimal):
    def __init__(self, scale=1, name="x"):
        super().__init__()
        self.scale, self.name = scale, name

    def load(self, path):
        return int(Path(path).read_text())


def test_config_chooses_the_model_its_options_and_the_initial_state(folder):
    (folder / "state.txt").write_text("7")
    (folder / "fit.yml").write_text(
        "model: landmarkerio.test.test_fit:FromConfig\n"
        "options: {scale: 3, name: y}\n"
        "initial_state: state.txt\n"
        "save_state: out.pkl\n"
        "from_saved: false\n"
        "learn_on_save: false\n"
        "min_images: 5\n"
    )
    fit = autofit_from_config({}, folder / "fit.yml")
    assert isinstance(fit.model, FromConfig) and (fit.model.scale, fit.model.name) == (
        3,
        "y",
    )
    assert (
        fit._initial == 7 and fit._fitter == 7
    )  # relative to the config, fits at once
    assert fit.save_path == folder / "out.pkl"
    assert (
        fit.from_saved is False and fit.learn_on_save is False and fit.min_images == 5
    )


@pytest.mark.parametrize(
    "text, message",
    [
        ("modle: x\n", "unknown settings"),
        ("model: nope\n", "not one of"),
        ("model: landmarkerio.fit:AutoFit\n", "not a subclass"),
        ("- a\n", "mapping"),
    ],
)
def test_a_bad_config_says_what_is_wrong(folder, text, message):
    (folder / "fit.yml").write_text(text)
    with pytest.raises((ValueError, TypeError), match=message):
        autofit_from_config({}, folder / "fit.yml")


def test_aam_component_limits_are_a_number_or_one_per_scale():
    pytest.importorskip("menpofit")
    from landmarkerio.fit_models import PatchAAM

    same = PatchAAM(max_shape_components=20, max_appearance_components="50")
    assert same.max_shape_components == 20 and same.max_appearance_components == 50
    per_scale = PatchAAM(
        max_shape_components=[15, 25.0], max_appearance_components=(75, 150)
    )
    assert per_scale.max_shape_components == [15, 25]
    assert per_scale.max_appearance_components == [75, 150]


def test_no_config_is_the_default_model():
    pytest.importorskip("menpofit")
    from landmarkerio.fit_models import PatchAAM

    assert isinstance(autofit_from_config({}).model, PatchAAM)


# ---- several models
def two_models(folder, **kwargs):
    ids = {p.name: p for p in sorted(folder.glob("*.png"))}
    proposals = Proposals(ids)
    kw = dict(background=False, proposals=proposals, **kwargs)
    return FitRegistry(
        {
            "detector": FakeAutoFit(
                ids, Detector(), initial_state=5, name="detector", **kw
            ),
            "aam": FakeAutoFit(ids, Full(), min_images=3, name="aam", **kw),
        }
    )


def test_a_registry_has_models_by_name_and_the_first_is_the_default(folder):
    reviewed(folder, 3)
    registry = two_models(folder)
    assert registry.get().name == "detector" and registry.get("aam").name == "aam"
    with pytest.raises(UnknownModel, match="no model 'nope'"):
        registry.get("nope")
    status = registry.status()
    assert status["name"] == "detector" and status["default"] == "detector"
    assert status["ready"] is True and status["models"]["aam"]["ready"] is False
    assert FitRegistry.of(registry) is registry
    assert FitRegistry.of(registry.get("aam")).default == "aam"


def test_saving_is_told_to_every_model_that_learns(folder):
    reviewed(folder, 3)
    registry = two_models(folder)
    registry.notify_saved("a0.png")
    assert registry.get("aam").status()["trained_on"] == 3
    assert registry.get("detector").status()["trained_on"] == 0  # cannot learn


def test_each_model_writes_its_own_proposals(folder):
    reviewed(folder, 3)
    write(folder, "bare", None)
    registry = two_models(folder)
    registry.notify_saved("")
    registry.get("detector").predict(SKELETON)
    registry.get("aam").predict(SKELETON)
    assert sorted(registry.proposals.models("bare.png")) == ["aam", "detector"]
    assert (
        registry.proposals.ids("aam")
        == registry.proposals.ids("detector")
        == ["bare.png"]
    )
    assert registry.get("aam").status()["proposals"] == 1


def test_the_prefit_the_config_asks_for_runs_at_start(folder):
    write(folder, "bare", None)
    registry = two_models(folder)
    assert registry.start_prefit(lambda t: SKELETON) is None  # not asked for
    registry.prefit_config = {"model": "detector", "template": "t"}
    job = registry.start_prefit(lambda t: SKELETON)
    assert job["state"] == "done" and registry.proposals.ids("detector") == ["bare.png"]
    registry.prefit_config = {"model": "detector", "template": "missing"}

    def broken(t):
        raise KeyError(t)

    assert registry.start_prefit(broken) is None  # logged, the server still starts


def test_config_with_several_models_and_a_prefit(folder):
    (folder / "fit.yml").write_text(
        "learn_on_save: false\n"
        "models:\n"
        "  - name: first\n"
        "    model: landmarkerio.test.test_fit:FromConfig\n"
        "    options: {scale: 2}\n"
        "  - model: landmarkerio.test.test_fit:FromConfig\n"
        "    name: second\n"
        "    min_images: 9\n"
        "    learn_on_save: true\n"
        "prefit: {model: second, template: face}\n"
    )
    registry = registry_from_config({}, folder / "fit.yml")
    assert list(registry.models) == ["first", "second"] and registry.default == "first"
    assert registry.get("first").model.scale == 2
    assert registry.get("second").min_images == 9
    # settings at the top are the defaults; a model can change them (FromConfig cannot
    # increment, so the switch stays off for it either way)
    assert registry.get("first").learn_on_save is False
    assert registry.prefit_config == {"model": "second", "template": "face"}
    assert registry.get("first").proposals is registry.get("second").proposals


@pytest.mark.parametrize(
    "text, message",
    [
        ("models: []\n", "list of models"),
        ("models: [{name: a}]\nmodel: patch-aam\n", "next to it"),
        ("models: [{nme: a}]\n", "unknown settings"),
        (
            "models: [{model: landmarkerio.test.test_fit:FromConfig},"
            " {model: landmarkerio.test.test_fit:FromConfig}]\n",
            "two models are called",
        ),
        (
            "model: landmarkerio.test.test_fit:FromConfig\nprefit: {model: x}\n",
            "needs a 'template'",
        ),
        (
            "model: landmarkerio.test.test_fit:FromConfig\nprefit: {model: x, template: t}\n",
            "prefit uses 'x'",
        ),
    ],
)
def test_a_bad_multi_model_config_says_what_is_wrong(folder, text, message):
    (folder / "fit.yml").write_text(text)
    with pytest.raises(ValueError, match=message):
        registry_from_config({}, folder / "fit.yml")


# ---- the API
@pytest.fixture
def api(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reviewed(folder, 3)
    write(folder, "new", [[10.0, 20.0]], reviewed=None)
    write(folder, "bare", None)
    cache = tmp_path / "cache"
    cache.mkdir()
    _, ids = cache_assets(
        "image", filepath_as_asset_id_under_dir(folder), folder, cache
    )
    fit = FakeAutoFit(ids, Full(), min_images=3, background=False)
    adapter = ReviewedLmAdapter(InplaceFileLmAdapter(ids), fit.notify_saved)
    app = serve_from_cache("image", cache, adapter, fit=fit)
    Sanic.test_mode = True
    return app.test_client, fit


BODY = {
    "version": 2,
    "labels": [{"label": "all", "mask": [0]}],
    "landmarks": {"points": [[10.0, 20.0]], "connectivity": []},
}


def test_api_status_and_fit(api):
    client, fit = api
    _, status = client.get("/api/v2/fit")
    assert status.json["ready"] is False and status.json["model"] == "Full"
    assert status.json["can_increment"] and status.json["unlabelled"] == 1
    _, response = client.post("/api/v2/fit/new.png/inplace", json=BODY)
    assert response.status == 409 and "needs 3 reviewed images" in response.text
    fit.refresh()
    _, response = client.post("/api/v2/fit/new.png/inplace", json=BODY)
    assert response.status == 200
    assert response.json["landmarks"]["points"] == [[13.0, 23.0]]
    assert response.json["labels"] == BODY["labels"]
    # the landmarks sent are the start, whatever is saved: unsaved changes are refined
    moved = {**BODY, "landmarks": {**BODY["landmarks"], "points": [[0.0, 0.0]]}}
    _, response = client.post("/api/v2/fit/new.png/inplace", json=moved)
    assert response.json["landmarks"]["points"] == [[3.0, 3.0]]
    # landmarks that are not placed: the model's own start
    partial = {**BODY, "landmarks": {**BODY["landmarks"], "points": [[None, None]]}}
    _, response = client.post("/api/v2/fit/bare.png/inplace", json=partial)
    assert response.status == 200 and response.json["landmarks"]["points"] == [
        [3.0, 3.0]
    ]
    # a box
    boxed = {**BODY, "box": [[10.0, 20.0], [30.0, 60.0]]}
    _, response = client.post("/api/v2/fit/new.png/inplace", json=boxed)
    assert response.json["landmarks"]["points"] == [[23.0, 43.0]]
    _, response = client.post("/api/v2/fit/nope/inplace", json=BODY)
    assert response.status == 404


def test_api_suggested_box(api):
    client, fit = api
    _, response = client.get("/api/v2/fit/box/bare.png")
    assert response.status == 409  # no model yet
    fit.refresh()
    _, response = client.get("/api/v2/fit/box/bare.png")
    assert response.status == 200 and response.json == [[0.0, 0.0], [0.0, 0.0]]
    _, response = client.get("/api/v2/fit/box/nope")
    assert response.status == 404


def test_api_train_and_predict_are_jobs(api):
    client, fit = api
    _, response = client.post("/api/v2/fit/train")
    assert response.status == 200 and response.json["state"] == "done"
    assert fit.status()["ready"]
    _, status = client.get("/api/v2/fit")
    assert status.json["job"]["kind"] == "train" and status.json["trained_on"] == 3
    # predict needs the layout of a template
    _, response = client.post("/api/v2/fit/predict/ibug68")
    assert response.status == 200 and response.json["kind"] == "predict"
    assert response.json["state"] == "done" and response.json["total"] == 1
    _, response = client.post("/api/v2/fit/predict/no_such_template")
    assert response.status == 404


def test_api_saving_through_the_server_teaches_the_model(api):
    client, fit = api
    fit.refresh()
    assert fit.status()["trained_on"] == 3
    time.sleep(0.01)
    payload = {**BODY, "landmarks": {**BODY["landmarks"], "points": [[7.0, 8.0]]}}
    _, response = client.put("/api/v2/landmarks/new.png/inplace", json=payload)
    assert response.status == 200, response.text
    assert fit.status()["trained_on"] == 4  # saved, so reviewed, so learnt from at once
    assert fit.model.log[-1] == ("add", ["new.png"])


def test_api_without_a_fit_service_has_no_fit_route(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write(folder, "a", [[1.0, 2.0]])
    cache = tmp_path / "cache"
    cache.mkdir()
    _, ids = cache_assets(
        "image", filepath_as_asset_id_under_dir(folder), folder, cache
    )
    app = serve_from_cache("image", cache, InplaceFileLmAdapter(ids))
    Sanic.test_mode = True
    _, response = app.test_client.get("/api/v2/fit")
    assert response.status == 404


def test_bare_fit_flag_means_the_default_config(monkeypatch, tmp_path):
    # "--fit" on its own parses to Path(""), which is "."; it is not a config to read
    from landmarkerio.scripts import lmio

    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    ns = lmio.build_argparser().parse_args(["image", str(tmp_path), "--fit"])
    assert ns.fit == Path()
    given = []

    class Stop(Exception):
        pass

    def fake(ids, config):
        given.append(config)
        raise Stop

    monkeypatch.setattr(lmio, "require_fit_dependencies", lambda: None)
    monkeypatch.setattr(lmio, "registry_from_config", fake)
    with pytest.raises(Stop):
        lmio.main(ns)
    assert given == [None]

    ns = lmio.build_argparser().parse_args(["image", str(tmp_path), "--fit", "fit.yml"])
    with pytest.raises(Stop):
        lmio.main(ns)
    assert given == [None, Path("fit.yml")]


def test_api_learn_one_saved_image_and_the_learn_on_save_switch(api):
    client, fit = api
    fit.learn_on_save = False
    client.post("/api/v2/fit/train")
    _, response = client.post("/api/v2/fit/learn/new.png")
    assert response.status == 409 and "saved" in response.json["message"]
    payload = {**BODY, "landmarks": {**BODY["landmarks"], "points": [[7.0, 8.0]]}}
    client.put("/api/v2/landmarks/new.png/inplace", json=payload)  # saved: reviewed
    assert fit.status()["trained_on"] == 3  # the switch is off, so it did not learn
    _, response = client.post("/api/v2/fit/learn/new.png")
    assert response.status == 200 and response.json["kind"] == "learn"
    assert fit.status()["trained_on"] == 4
    _, response = client.post("/api/v2/fit/learn/nope.png")
    assert response.status == 404
    _, response = client.post("/api/v2/fit/learn_on_save", json={"enabled": True})
    assert response.status == 200 and response.json["learn_on_save"] is True
    _, response = client.post("/api/v2/fit/learn_on_save", json={"enabled": "yes"})
    assert response.status == 400


@pytest.fixture
def api2(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    reviewed(folder, 3)
    write(folder, "new", [[10.0, 20.0]], reviewed=None)
    write(folder, "bare", None)
    cache = tmp_path / "cache"
    cache.mkdir()
    _, ids = cache_assets(
        "image", filepath_as_asset_id_under_dir(folder), folder, cache
    )
    proposals = Proposals(ids)
    kw = dict(background=False, proposals=proposals)
    registry = FitRegistry(
        {
            "detector": FakeAutoFit(
                ids, Full(), initial_state=[1], name="detector", **kw
            ),
            "aam": FakeAutoFit(ids, Full(), min_images=3, name="aam", **kw),
        }
    )
    adapter = ReviewedLmAdapter(
        InplaceFileLmAdapter(ids), registry.notify_saved, registry.proposals
    )
    app = serve_from_cache("image", cache, adapter, fit=registry)
    Sanic.test_mode = True
    return app.test_client, registry


def test_api_models_by_name(api2):
    client, registry = api2
    _, status = client.get("/api/v2/fit")
    assert status.json["name"] == "detector" and set(status.json["models"]) == {
        "detector",
        "aam",
    }
    # the default model is the one without a name in the route
    _, response = client.post("/api/v2/fit/new.png/inplace", json=BODY)
    assert response.json["landmarks"]["points"] == [[11.0, 21.0]]
    assert response.json["fitted_by"] == "detector"
    # a model that is not ready says so, the other one is not affected
    _, response = client.post("/api/v2/fit/models/aam/fit/new.png/inplace", json=BODY)
    assert (
        response.status == 409 and "needs 3 reviewed images" in response.json["message"]
    )
    _, response = client.post("/api/v2/fit/models/aam/train")
    assert response.status == 200 and response.json["kind"] == "train"
    _, response = client.post("/api/v2/fit/models/aam/fit/new.png/inplace", json=BODY)
    assert response.json["landmarks"]["points"] == [[13.0, 23.0]]
    assert response.json["fitted_by"] == "aam"
    _, response = client.get("/api/v2/fit/models/aam/box/bare.png")
    assert response.status == 200
    _, response = client.post("/api/v2/fit/models/nope/train")
    assert response.status == 404 and "no model 'nope'" in response.json["message"]
    _, response = client.post("/api/v2/fit/models/aam/fit/nope/inplace", json=BODY)
    assert response.status == 404
    _, response = client.post(
        "/api/v2/fit/models/aam/learn_on_save", json={"enabled": False}
    )
    assert response.status == 200 and response.json["learn_on_save"] is False
    assert registry.get("detector").learn_on_save is True  # the other is not changed


def test_api_prefit_then_accept_as_it_is(api2):
    client, registry = api2
    # nothing there to serve
    _, response = client.get("/api/v2/landmarks")
    assert "bare.png" not in response.json
    _, response = client.post(
        "/api/v2/fit/models/detector/predict/ibug68", json={"overwrite": True}
    )
    assert response.status == 200 and response.json["total"] == 1
    _, response = client.post("/api/v2/fit/models/nope/predict/ibug68")
    assert response.status == 404
    _, response = client.post("/api/v2/fit/models/aam/predict/ibug68")
    assert response.status == 409  # the aam has no model yet
    # the landmarks of the image are now the proposal
    _, response = client.get("/api/v2/landmarks")
    assert response.json["bare.png"] == ["inplace"]
    _, response = client.get("/api/v2/landmarks/bare.png/inplace")
    assert (
        response.json["reviewed"] is False
        and response.json["source"] == "prefit:detector"
    )
    assert registry.get("aam").status()["unlabelled"] == 1  # still unlabelled
    # accepted with one save and no edits
    _, response = client.put("/api/v2/landmarks/bare.png/inplace", json=response.json)
    assert response.status == 200
    _, response = client.get("/api/v2/landmarks/bare.png/inplace")
    assert (
        response.json["reviewed"] is True and response.json["fitted_by"] == "detector"
    )
    assert "source" not in response.json
    assert registry.get("aam").status()["annotated"] == 4
