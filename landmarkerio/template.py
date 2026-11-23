import abc
import itertools
import os.path as p
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml
from loguru import logger

from landmarkerio import TEMPLATE_DINAME, FileExt
from landmarkerio.types import PathLike

DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "default_templates"


class Group(NamedTuple):
    label: str
    n: int
    indices: Sequence[Tuple[int, int]]
    # Points that stay where they are when the group is resampled (indices within the group)
    anchors: Sequence[int] = ()
    # For some of the other points, the fraction of the arc length between the anchors around it at
    # which it goes when resampling (indices within the group); equal steps without it
    spacing: Optional[Dict[int, float]] = None


class MissingTemplate(ValueError):
    def __init__(self, template_id: str) -> None:
        super().__init__(f"Cannot find template with id '{template_id}'")


def parse_connectivity(index_lst: Sequence[str], n: int) -> Sequence[Tuple[int, int]]:
    index: List[Tuple[int, int]] = []
    for i in index_lst:
        if ":" in i:
            # User is providing a slice
            start, end = (int(x) for x in i.split(":"))
            index.extend((x, x + 1) for x in range(start, end))
        else:
            # Just a standard pair of numbers
            start, end = (int(j) for j in i.split(" "))
            index.append((start, end))

    indexes = set(itertools.chain.from_iterable(index))
    if index and (min(indexes) < 0 or max(indexes) > n):
        raise ValueError("Invalid connectivity")

    return index


def parse_anchors(label: str, n: int, anchors: Any) -> Sequence[int]:
    if anchors is None:
        return ()
    if not isinstance(anchors, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in anchors
    ):
        raise ValueError(
            f"anchors of '{label}' must be a list of point indices between 0 and {n - 1}"
        )
    return tuple(anchors)


def parse_spacing(
    label: str, n: int, spacing: Any, anchors: Sequence[int]
) -> Optional[Dict[int, float]]:
    if spacing is None:
        return None
    if not isinstance(spacing, dict):
        raise ValueError(f"spacing of '{label}' must map point indices to fractions")
    result: Dict[int, float] = {}
    for key, fraction in spacing.items():
        if (
            not isinstance(key, int)
            or not 0 <= key < n
            or key in anchors
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                f"spacing of '{label}' must give a fraction between 0 and 1 to points "
                f"of the group that are not anchors; got {key}: {fraction}"
            )
        result[key] = float(fraction)
    return result


def load_yaml_template(filepath: PathLike, n_dims: int):
    with Path(filepath).open("r") as f:
        data = yaml.safe_load(f)

    if "groups" in data:
        raw_groups = data["groups"]
    else:
        raise KeyError(f"Missing 'groups' or 'template' key in yaml file {filepath}")

    groups = []
    for k, group in enumerate(raw_groups):
        label = group.get("label", str(k))  # Allow simple ordered groups

        n = group["points"]  # Should raise KeyError by design if missing
        connectivity = group.get("connectivity", [])

        if isinstance(connectivity, str):
            # a string is a Sequence too, so it has to be dealt with first
            if connectivity.strip() == "cycle":
                connectivity = [f"0:{n - 1}", f"{n - 1} 0"]
            else:
                connectivity = [
                    connectivity
                ]  # 'connectivity: 0:3' for 'connectivity: [0:3]'
        if isinstance(connectivity, Sequence):
            indices = parse_connectivity(connectivity, n)
        else:
            indices = []  # Couldn't parse connectivity, safe default

        anchors = parse_anchors(label, n, group.get("anchors"))
        spacing = parse_spacing(label, n, group.get("spacing"), anchors)
        groups.append(Group(label, n, indices, anchors, spacing))

    return build_json(groups, n_dims)


def parse_group(group):
    # split on \n and strip left and right whitespace.
    x = [l.strip() for l in group.split("\n")]
    label, n_str = x[0].split(" ")
    n = int(n_str)
    index_str = x[1:]
    if len(index_str) == 0:
        return Group(label, n, [])
    index = parse_connectivity(index_str, n)
    return Group(label, n, index)


def group_to_json(group, n_dims):
    group_json = {}
    lms = [{"point": [None] * n_dims}] * group.n
    group_json["landmarks"] = lms
    group_json["connectivity"] = group.indices
    group_json["label"] = group.label
    return group_json


def build_json(groups: Sequence[Group], n_dims: int) -> Dict[str, Any]:
    n_points = sum(g.n for g in groups)
    offset = 0
    connectivity = []
    labels = []
    for g in groups:
        connectivity += [[j + offset for j in i] for i in g.indices]
        entry: Dict[str, Any] = {
            "label": g.label,
            "mask": list(range(offset, offset + g.n)),
        }
        if g.anchors:
            entry["anchors"] = [i + offset for i in g.anchors]
        if g.spacing:
            # JSON keys are strings
            entry["spacing"] = {str(i + offset): f for i, f in g.spacing.items()}
        labels.append(entry)
        offset += g.n

    lm_json = {
        "labels": labels,
        "landmarks": {
            "connectivity": connectivity,
            "points": [[None] * n_dims] * n_points,
        },
        "version": 2,
    }

    return lm_json


def group_to_dict(g: Group) -> Dict[str, Any]:
    data: Dict[str, Any] = {"label": g.label, "points": g.n}
    if g.indices:
        data["connectivity"] = [f"{c[0]} {c[1]}" for c in g.indices]
    if g.anchors:
        data["anchors"] = list(g.anchors)
    if g.spacing:
        data["spacing"] = dict(g.spacing)
    return data


def load_template(path: PathLike, n_dims: int) -> Dict[str, Any]:
    return load_yaml_template(path, n_dims)


class TemplateAdapter(abc.ABC):
    r"""
    Abstract definition of an adapter that can be passed to app_for_adapter in
    order to generate a legal Flask implementation of landmarker.io's REST API
    for Template retrieval.
    """

    @abc.abstractmethod
    def template_ids(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def load_template(self, lm_id: str):
        pass


class FileTemplateAdapter(TemplateAdapter):
    def __init__(self, n_dims: int, template_dir: Optional[PathLike] = None) -> None:
        self.n_dims = n_dims
        if template_dir is None:
            # try the user folder, then the templates that ship with the package
            user_templates = Path(p.expanduser(p.join("~", TEMPLATE_DINAME)))
            if user_templates.exists():
                template_dir = user_templates
            else:
                template_dir = DEFAULT_TEMPLATE_DIR
                logger.info(
                    "no template dir given and {} does not exist: using the "
                    "templates that ship with landmarkerio",
                    user_templates,
                )
        self.template_dir = Path(p.abspath(p.expanduser(template_dir)))
        logger.debug("templates directory: {}", self.template_dir)

    def template_ids(self) -> Sequence[str]:
        return [t.stem for t in self.template_paths()]

    def template_paths(self) -> Sequence[Path]:
        # .yml, or .yaml; the .yml wins if both exist
        paths = {}
        for ext in (".yaml", FileExt.template):
            for path in self.template_dir.glob("*" + ext):
                paths[path.stem] = path
        return sorted(paths.values())

    def load_template(self, lm_id: str):
        for path in self.template_paths():
            if path.stem == lm_id:
                return load_template(path, self.n_dims)
        raise MissingTemplate(lm_id)


class CachedFileTemplateAdapter(FileTemplateAdapter):
    def __init__(
        self,
        n_dims: int,
        template_dir: Optional[PathLike] = None,
    ) -> None:
        super().__init__(n_dims, template_dir=template_dir)

        self._cache = {
            lm_id: FileTemplateAdapter.load_template(self, lm_id)
            for lm_id in FileTemplateAdapter.template_ids(self)
        }
        logger.debug(
            "cached {} templates ({})", len(self._cache), ", ".join(self._cache.keys())
        )

    def load_template(self, lm_id: str):
        try:
            return self._cache[lm_id]
        except KeyError:
            raise MissingTemplate(lm_id)
