import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sanic import Sanic

from landmarkerio.fit import AutoFit
from landmarkerio.landmark import InplaceFileLmAdapter, ReviewedLmAdapter
from landmarkerio.servers.serve import serve_from_cache


class FakeFit(AutoFit):
    """AutoFit with the menpofit parts replaced, to test when it learns and what it serves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def _load(self, asset_id):
        return asset_id

    def _build_model(self, images):
        self.log.append(("build", sorted(images)))
        return list(images)

    def _increment(self, model, images):
        self.log.append(("add", sorted(images)))
        return model + list(images)

    def _make_fitter(self, model):
        return len(model)

    def _fit(self, fitter, asset_id, start, max_iters):
        return start + fitter  # moved by the number of images the model has seen


def write(folder: Path, name: str, points, reviewed=True) -> None:
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(folder / f"{name}.png")
    data = {
        "version": 2,
        "labels": [{"label": "all", "mask": list(range(len(points)))}],
        "landmarks": {"points": points, "connectivity": []},
    }
    if reviewed:
        data["reviewed"] = True
    (folder / f"{name}.ljson").write_text(json.dumps(data))


@pytest.fixture
def folder(tmp_path):
    return tmp_path


def make(folder, **kwargs) -> FakeFit:
    ids = {p.stem: p for p in sorted(folder.glob("*.png"))}
    return FakeFit(ids, background=False, **kwargs)


def test_no_model_before_enough_reviewed_images(folder):
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0], [3.0, 4.0]])
    fit = make(folder, min_images=4)
    fit.refresh()
    assert fit.status()["ready"] is False
    assert fit.status()["annotated"] == 3
    with pytest.raises(Exception, match="needs 4 reviewed images"):
        fit.fit("a0")


def test_only_reviewed_images_count_unless_asked(folder):
    write(folder, "r", [[1.0, 2.0]])
    write(folder, "u", [[1.0, 2.0]], reviewed=False)
    write(folder, "partial", [[1.0, None]])
    assert make(folder).annotated_ids() == ["r"]
    assert make(folder, from_saved=False).annotated_ids() == [
        "r",
        "u",
    ]  # never the incomplete one


def test_builds_then_adds_new_images_to_the_model(folder):
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0]])
    fit = make(folder, min_images=3)
    fit.refresh()
    assert fit.log == [("build", ["a0", "a1", "a2"])]
    fit.refresh()  # nothing new: nothing done
    assert len(fit.log) == 1
    write(folder, "a3", [[1.0, 2.0]])
    fit.ids_to_paths["a3"] = folder / "a3.png"
    fit.refresh()
    assert fit.log[-1] == ("add", ["a3"])  # incremental, not from scratch
    assert fit.status()["trained_on"] == 4


def test_a_resaved_image_that_was_learnt_from_rebuilds(folder):
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0]])
    fit = make(folder, min_images=3)
    fit.refresh()
    path = folder / "a1.ljson"
    later = os.stat(path).st_mtime + 10
    write(folder, "a1", [[9.0, 9.0]])
    os.utime(path, (later, later))
    fit.refresh()
    assert fit.log[-1] == (
        "build",
        ["a0", "a1", "a2"],
    )  # an incremental model can't forget


def test_fit_refines_the_landmarks_it_is_given(folder):
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0]])
    write(folder, "new", [[10.0, 20.0]], reviewed=False)
    fit = make(folder, min_images=3)
    fit.refresh()
    assert fit.fit("new").tolist() == [[13.0, 23.0]]
    # no landmarks to start from
    (folder / "new.ljson").unlink()
    with pytest.raises(ValueError, match="no landmarks to refine"):
        fit.fit("new")


def test_background_learning_finishes_and_a_failure_is_reported(folder):
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0]])
    ids = {p.stem: p for p in folder.glob("*.png")}
    fit = FakeFit(ids, min_images=3, background=True)
    fit.notify_saved("a0")
    fit._executor.shutdown(wait=True)
    assert fit.status()["ready"] is True

    class Broken(FakeFit):
        def _build_model(self, images):
            raise RuntimeError("boom")

    broken = Broken(ids, min_images=3, background=True)
    broken.notify_saved("a0")
    broken._executor.shutdown(wait=True)
    assert "boom" in broken.status()["error"]


def test_saving_marks_reviewed_and_tells_the_fitter(folder):
    write(folder, "a", [[1.0, 2.0]], reviewed=False)
    told = []
    adapter = ReviewedLmAdapter(
        InplaceFileLmAdapter({"a": folder / "a.png"}), told.append
    )
    adapter.save_landmark(
        "a", "inplace", {"version": 2, "landmarks": {"points": [[5.0, 6.0]]}}
    )
    assert json.loads((folder / "a.ljson").read_text())["reviewed"] is True
    assert told == ["a"]


@pytest.fixture
def api(folder, tmp_path, monkeypatch):
    from landmarkerio.cache import (
        cache_assets,
        filepath_as_asset_id_under_dir,
    )

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for i in range(3):
        write(folder, f"a{i}", [[1.0, 2.0]])
    write(folder, "new", [[10.0, 20.0]], reviewed=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    _, ids = cache_assets(
        "image", filepath_as_asset_id_under_dir(folder), folder, cache
    )
    fit = FakeFit(ids, min_images=3, background=False)
    adapter = ReviewedLmAdapter(InplaceFileLmAdapter(ids), fit.notify_saved)
    app = serve_from_cache("image", cache, adapter, fit=fit)
    Sanic.test_mode = True
    return app.test_client, fit


def test_api_status_and_fit(api):
    client, fit = api
    _, status = client.get("/api/v2/fit")
    assert status.json["ready"] is False
    body = {
        "version": 2,
        "labels": [{"label": "all", "mask": [0]}],
        "landmarks": {"points": [[10.0, 20.0]], "connectivity": []},
    }
    _, response = client.post("/api/v2/fit/new.png/inplace", json=body)
    assert response.status == 409 and "needs 3 reviewed images" in response.text
    fit.refresh()
    _, response = client.post("/api/v2/fit/new.png/inplace", json=body)
    assert response.status == 200
    assert response.json["landmarks"]["points"] == [[13.0, 23.0]]
    assert response.json["labels"] == body["labels"]
    # the landmarks sent are the start, whatever is saved: unsaved changes are refined
    moved = {**body, "landmarks": {**body["landmarks"], "points": [[0.0, 0.0]]}}
    _, response = client.post("/api/v2/fit/new.png/inplace", json=moved)
    assert response.json["landmarks"]["points"] == [[3.0, 3.0]]
    # landmarks that are not all placed are nothing to refine
    partial = {**body, "landmarks": {**body["landmarks"], "points": [[None, None]]}}
    _, response = client.post("/api/v2/fit/new.png/inplace", json=partial)
    assert response.status == 409 and "no landmarks to refine" in response.text
    _, response = client.post("/api/v2/fit/nope/inplace", json=body)
    assert response.status == 404


def test_api_saving_through_the_server_teaches_the_model(api):
    client, fit = api
    fit.refresh()
    assert fit.status()["trained_on"] == 3
    time.sleep(0.01)
    payload = {
        "version": 2,
        "labels": [{"label": "all", "mask": [0]}],
        "landmarks": {"points": [[7.0, 8.0]], "connectivity": []},
    }
    _, response = client.put("/api/v2/landmarks/new.png/inplace", json=payload)
    assert response.status == 200, response.text
    assert fit.status()["trained_on"] == 4  # saved, so reviewed, so learnt from at once
    assert fit.log[-1] == ("add", ["new.png"])


def test_api_without_a_fit_service_has_no_fit_route(folder, tmp_path, monkeypatch):
    from landmarkerio.cache import cache_assets, filepath_as_asset_id_under_dir

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
