r"""
The models that ``lmio --fit-model`` knows by name. They are menpofit Active Appearance Models,
which can be grown with ``increment``; menpofit is imported only when one is used.
"""

import copy
from typing import Any, Dict, Sequence

import numpy as np

from landmarkerio.fit import LJSON_GROUP, FitModel


def _per_scale(value: Any) -> Any:
    r"""A number for every scale, or a list with one for each scale (coarse to fine)."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return int(value)


class _AAMModel(FitModel):
    r"""Shared by the AAMs: grow a copy, fit with Lucas-Kanade."""

    #: menpofit aligns the shapes of the images it is given, which needs two
    min_increment = 2

    def __init__(
        self,
        diagonal: float = 150,
        scales: Sequence[float] = (0.5, 1.0),
        max_shape_components: Any = 20,
        max_appearance_components: Any = 100,
        max_iters: Sequence[int] = (15, 10),
        min_images: int = 8,
        initial_scale: float = 0.6,
    ) -> None:
        self.diagonal = float(diagonal)
        self.scales = tuple(scales)
        self.max_shape_components = _per_scale(max_shape_components)
        self.max_appearance_components = _per_scale(max_appearance_components)
        self.max_iters = list(max_iters)
        self.min_images = int(min_images)
        self.initial_scale = float(initial_scale)

    def prepare(self, image, training):
        if image.n_channels == 3:
            image = image.as_greyscale()
        if training:
            # smaller images are cheaper to learn from, and the fitter works on any size
            image = image.crop_to_landmarks_proportion(0.2, group=LJSON_GROUP)
            diagonal = image.diagonal()
            if diagonal > 400:
                image = image.rescale(400.0 / diagonal)
        return image

    def increment(self, state, images):
        grown = copy.deepcopy(state)  # a fit may be using the current state
        grown.increment(images, group=LJSON_GROUP, verbose=False)
        return grown

    def fitter(self, state):
        from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

        return LucasKanadeAAMFitter(state, lk_algorithm_cls=WibergInverseCompositional)

    def initialize(self, state, image, box=None):
        r"""
        The mean shape of the model in a box: the one that was drawn, or the middle of the image
        (``initial_scale`` of its shorter side). The shape is scaled to fill the box, keeping its
        proportions, and centred in it.
        """
        mean = state.shape_models[-1].model.mean().points
        extent = mean.max(axis=0) - mean.min(axis=0)
        if box is None:
            side = self.initial_scale * min(image.shape)
            centre = np.array(image.shape) / 2.0
            size = np.array([side, side])
        else:
            centre, size = box.mean(axis=0), box[1] - box[0]
        scale = (size / extent).min()
        return (mean - mean.mean(axis=0)) * scale + centre

    def fit(self, fitter, image, start):
        from menpo.shape import PointCloud

        result = fitter.fit_from_shape(
            image, PointCloud(start), max_iters=self.max_iters
        )
        return np.asarray(result.final_shape.points)


class PatchAAM(_AAMModel):
    r"""
    A patch-based AAM on image gradients (``igo``): the default. It models patches around each
    landmark, so it copes with lighting and works on faces and on most things with clear parts.
    """

    def __init__(self, patch_sizes: Sequence[int] = (15, 23), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.patch_shape = [(int(p), int(p)) for p in patch_sizes]

    def build(self, images):
        from menpo.feature import igo
        from menpofit.aam import PatchAAM as MenpoPatchAAM

        return MenpoPatchAAM(
            images,
            group=LJSON_GROUP,
            patch_shape=self.patch_shape,
            diagonal=self.diagonal,
            scales=self.scales,
            holistic_features=igo,
            max_shape_components=self.max_shape_components,
            max_appearance_components=self.max_appearance_components,
            verbose=False,
        )


class HolisticAAM(_AAMModel):
    r"""A holistic AAM: the whole warped appearance (``igo`` features), for small, rigid shapes."""

    def build(self, images):
        from menpo.feature import igo
        from menpofit.aam import HolisticAAM as MenpoHolisticAAM

        return MenpoHolisticAAM(
            images,
            group=LJSON_GROUP,
            diagonal=self.diagonal,
            scales=self.scales,
            holistic_features=igo,
            max_shape_components=self.max_shape_components,
            max_appearance_components=self.max_appearance_components,
            verbose=False,
        )


MODELS: Dict[str, type] = {"patch-aam": PatchAAM, "holistic-aam": HolisticAAM}
