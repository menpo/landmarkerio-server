from pathlib import Path

import pytest
import yaml

from landmarkerio.template import DEFAULT_TEMPLATE_DIR, load_yaml_template

TEMPLATE = """
groups:
  - label: jaw
    points: 5
    connectivity: ['0:4']
    anchors: [0, 2, 4]
    spacing: {1: 0.4, 3: 0.6}
  - label: eye
    points: 4
    connectivity: ['0:3', '3 0']
    anchors: [0]
  - label: dot
    points: 1
"""


def load(tmp_path: Path, text: str, n_dims: int = 2):
    path = tmp_path / "t.yml"
    path.write_text(text)
    return load_yaml_template(path, n_dims)


def test_anchors_and_spacing_are_global_in_the_labels(tmp_path):
    labels = load(tmp_path, TEMPLATE)["labels"]
    assert labels[0] == {
        "label": "jaw",
        "mask": [0, 1, 2, 3, 4],
        "anchors": [0, 2, 4],
        "spacing": {"1": 0.4, "3": 0.6},
    }
    # the offsets of the groups are added
    assert labels[1] == {"label": "eye", "mask": [5, 6, 7, 8], "anchors": [5]}
    # a group without them has neither key (and old clients and menpo ignore them anyway)
    assert labels[2] == {"label": "dot", "mask": [9]}


@pytest.mark.parametrize(
    "extra",
    [
        "anchors: [9]",
        "anchors: [-1]",
        "anchors: [0.5]",
        "anchors: 3",
        "anchors: [true]",
        "spacing: {1: 1.5}",
        "spacing: {1: 0}",
        "spacing: {7: 0.5}",
        "spacing: [0.5]",
        "anchors: [2]\n    spacing: {2: 0.5}",
    ],
)
def test_nonsense_anchors_and_spacing_are_errors(tmp_path, extra):
    text = f"groups:\n  - label: a\n    points: 5\n    connectivity: ['0:4']\n    {extra}\n"
    with pytest.raises(ValueError):
        load(tmp_path, text)


def test_a_template_without_them_is_as_before(tmp_path):
    labels = load(
        tmp_path, "groups:\n  - label: a\n    points: 3\n    connectivity: ['0:2']\n"
    )["labels"]
    assert labels == [{"label": "a", "mask": [0, 1, 2]}]


def test_the_shipped_ibug68_template_says_what_to_keep_and_how_to_space_the_rest():
    template = load_yaml_template(DEFAULT_TEMPLATE_DIR / "ibug68.yml", 2)
    assert len(template["landmarks"]["points"]) == 68
    groups = yaml.safe_load((DEFAULT_TEMPLATE_DIR / "ibug68.yml").read_text())["groups"]
    for group, label in zip(groups, template["labels"]):
        n, anchors = group["points"], group["anchors"]
        spacing = group["spacing"]
        assert label["anchors"] == [i + label["mask"][0] for i in anchors]
        # every point that is not an anchor has a fraction, and anchors have none
        assert sorted(set(range(n)) - set(anchors)) == sorted(spacing)
        # between two anchors the fractions increase
        run = []
        for i in range(n):
            if i in anchors:
                run = []
            else:
                assert all(spacing[i] > spacing[j] for j in run), (group["label"], i)
                run.append(i)
    # the corners of a face
    jaw, nose, mouth = groups[0], groups[3], groups[6]
    assert jaw["anchors"] == [0, 8, 16]
    assert [3, 4] == [
        a for a in nose["anchors"] if a in (3, 4)
    ]  # 30 and 31, either side of the corner
    assert {0, 3, 6, 9} <= set(mouth["anchors"])  # 48, 51, 54, 57


def test_connectivity_as_a_single_string_and_cycle(tmp_path):
    # 'connectivity: 0:3' (not a list) and 'connectivity: cycle' are what the documentation says
    text = (
        "groups:\n"
        "  - label: chain\n    points: 4\n    connectivity: 0:3\n"
        "  - label: ring\n    points: 4\n    connectivity: cycle\n"
        "  - label: pair\n    points: 2\n    connectivity: 0 1\n"
    )
    edges = load(tmp_path, text)["landmarks"]["connectivity"]
    assert edges == [
        [0, 1], [1, 2], [2, 3],  # the chain
        [4, 5], [5, 6], [6, 7], [7, 4],  # the ring, closed
        [8, 9],
    ]  # fmt: skip


def test_templates_may_end_in_yaml(tmp_path):
    from landmarkerio.template import FileTemplateAdapter, MissingTemplate

    body = "groups:\n  - label: a\n    points: 2\n"
    (tmp_path / "heart.yaml").write_text(body)
    (tmp_path / "face.yml").write_text(body)
    (tmp_path / "both.yaml").write_text(body)
    (tmp_path / "both.yml").write_text(body.replace("points: 2", "points: 3"))
    (tmp_path / "notes.txt").write_text("not a template")
    adapter = FileTemplateAdapter(2, template_dir=tmp_path)
    assert sorted(adapter.template_ids()) == ["both", "face", "heart"]
    assert len(adapter.load_template("heart")["landmarks"]["points"]) == 2
    assert len(adapter.load_template("both")["landmarks"]["points"]) == 3  # .yml wins
    with pytest.raises(MissingTemplate):
        adapter.load_template("notes")
