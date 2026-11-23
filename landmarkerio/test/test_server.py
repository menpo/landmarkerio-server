import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sanic import Sanic

from landmarkerio.cache import (
    cache_assets,
    ensure_cache_dir,
    filepath_as_asset_id_under_dir,
)
from landmarkerio.landmark import InplaceFileLmAdapter, SeparateDirFileLmAdapter
from landmarkerio.servers.serve import serve_from_cache


def make_images(folder: Path, n: int = 3) -> None:
    rng = np.random.RandomState(0)
    for i in range(n):
        pixels = rng.randint(0, 255, size=(40, 30, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"face_{i}.png")


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A test client for a server over three small PNGs; the user has no ~/.lmiotemplates."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no templates of the user's own
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assets = tmp_path / "assets"
    assets.mkdir()
    make_images(assets)
    cache = tmp_path / "cache"
    cache.mkdir()
    _, ids_to_paths = cache_assets(
        "image", filepath_as_asset_id_under_dir(assets), assets, cache
    )
    app = serve_from_cache("image", cache, InplaceFileLmAdapter(ids_to_paths))
    Sanic.test_mode = True
    return app.test_client, ids_to_paths


def get(client, path, **kwargs):
    _, response = client.get(f"/api/v2/{path}", **kwargs)
    return response


def test_mode(served):
    client, _ = served
    assert get(client, "mode").json == "image"


def test_collection_lists_the_assets(served):
    client, ids = served
    assert get(client, "collections").json == ["all"]
    assert sorted(get(client, "collections/all").json) == sorted(ids)


def test_images_lists_the_assets(served):
    # the adapter used to look for an image.json that is no longer written: always empty
    client, ids = served
    assert sorted(get(client, "images").json) == sorted(ids)


def test_texture_and_thumbnail_are_jpegs(served):
    client, ids = served
    asset_id = sorted(ids)[0]
    for kind in ("textures", "thumbnails"):
        response = get(client, f"{kind}/{asset_id}")
        assert response.status == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.body[:2] == b"\xff\xd8"


def test_missing_asset_is_a_404(served):
    client, _ = served
    assert get(client, "textures/nothing.png").status == 404


def test_the_templates_that_ship_with_the_package_are_used(served):
    client, _ = served
    assert get(client, "templates").json == ["ibug68"]
    template = get(client, "templates/ibug68").json
    assert len(template["landmarks"]["points"]) == 68
    assert [g["label"] for g in template["labels"]][:2] == ["jaw", "left_eyebrow"]


def test_landmarks_start_as_the_template_and_round_trip(served):
    client, ids = served
    asset_id = sorted(ids)[0]
    empty = get(client, f"landmarks/{asset_id}/ibug68").json
    assert all(point == [None, None] for point in empty["landmarks"]["points"])

    placed = json.loads(json.dumps(empty))
    placed["landmarks"]["points"] = [[float(i), 2.0 * i] for i in range(68)]
    _, response = client.put(f"/api/v2/landmarks/{asset_id}/ibug68", json=placed)
    assert response.status == 200

    again = get(client, f"landmarks/{asset_id}/ibug68").json
    assert again["landmarks"]["points"] == placed["landmarks"]["points"]
    # they are kept next to the asset, as .ljson
    assert Path(ids[asset_id]).with_suffix(".ljson").exists()
    assert get(client, "landmarks").json == {asset_id: ["inplace"]}


def test_a_page_on_the_internet_may_reach_a_server_on_this_machine(served):
    # the preflight of the hosted landmarker.io asks to reach a private network
    client, ids = served
    asset_id = sorted(ids)[0]
    _, response = client.options(
        f"/api/v2/landmarks/{asset_id}/ibug68",
        headers={
            "Origin": "https://menpo.github.io",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert response.headers["access-control-allow-private-network"] == "true"
    assert response.headers["access-control-allow-origin"] in (
        "*",
        "https://menpo.github.io",
    )


def test_cache_and_landmark_directories_are_created(tmp_path):
    # both used to create the directory only when it already existed
    cache = ensure_cache_dir(tmp_path / "new" / "cache")
    assert cache.is_dir()
    adapter = SeparateDirFileLmAdapter(tmp_path / "new" / "landmarks")
    assert adapter.lm_dir.is_dir()


def test_jpeg_textures_from_png_assets_are_cached_in_a_fresh_process(tmp_path):
    # menpo < 0.13 could not export a JPEG after importing a PNG (menpo #904): a new
    # interpreter, because the state of Pillow is global
    import subprocess
    import sys

    assets = tmp_path / "assets"
    assets.mkdir()
    make_images(assets, 1)
    code = (
        "import sys; from pathlib import Path; "
        "from landmarkerio.cache import cache_assets, filepath_as_asset_id_under_dir; "
        "a = Path(sys.argv[1]); c = Path(sys.argv[2]); c.mkdir(); "
        "print(sorted(cache_assets('image', filepath_as_asset_id_under_dir(a), a, c)[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(assets), str(tmp_path / "cache")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "face_0.png" in result.stdout
