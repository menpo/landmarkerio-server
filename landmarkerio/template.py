import abc
import itertools
import os.path as p
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml
from flask import safe_join
from loguru import logger

from landmarkerio import FileExt, TEMPLATE_DINAME
from landmarkerio.types import PathLike


class Group(NamedTuple):
    label: str
    n: int
    indices: Sequence[Tuple[int, int]]


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

        if isinstance(connectivity, Sequence):
            indices = parse_connectivity(connectivity, n)
        elif connectivity == "cycle":
            indices = parse_connectivity(["0:{n - 1}", "{n - 1} 0"], n)
        else:
            indices = []  # Couldn't parse connectivity, safe default

        groups.append(Group(label, n, indices))

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
        labels.append({"label": g.label, "mask": list(range(offset, offset + g.n))})
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
    data = {"label": g.label, "points": g.n}
    if g.indices:
        data["connectivity"] = [f"{c[0]} {c[1]}" for c in g.indices]
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
            # try the user folder
            user_templates = Path(p.expanduser(p.join("~", TEMPLATE_DINAME)))
            if user_templates.exists():
                template_dir = user_templates
            else:
                raise ValueError(
                    f"No template dir provided and {user_templates} doesn't exist"
                )
        self.template_dir = Path(p.abspath(p.expanduser(template_dir)))
        logger.debug("templates directory: {}", self.template_dir)

    def template_ids(self) -> Sequence[str]:
        return [t.stem for t in self.template_paths()]

    def template_paths(self) -> Sequence[Path]:
        return list(self.template_dir.glob("*" + FileExt.template))

    def load_template(self, lm_id: str):
        fp = safe_join(str(self.template_dir), lm_id + FileExt.template)
        return load_template(fp, self.n_dims)


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
