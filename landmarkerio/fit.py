import cPickle
import json
import os
import shutil
import base64
import uuid
import os.path as p
from io import BytesIO
from pathlib import Path
from collections import OrderedDict

from PIL import Image as PILImage
import numpy as np

import menpo.io as mio
from menpo.feature import fast_dsift
from menpo.image import MaskedImage
from menpofit.aam import LucasKanadeAAMFitter
from menpo.transform import AlignmentSimilarity
from menpofit.aam import AAMBuilder
from menpo.io.input.landmark import _parse_ljson_v2
from menpo.landmark.base import LandmarkGroup
from menpodetect import load_dlib_frontal_face_detector

IMG_FILE = u'img.pickle'
LMS_FILE = u'landmarks.ljson'
CACHE_DIR = u'.cache'


def _model_directory(source_dir, name):
    return p.join(p.abspath(p.expanduser(source_dir)), name)


def _parse_ljson(lms):
    lms_dict = json.loads(lms, object_pairs_hook=OrderedDict)
    pointcloud, labels_to_masks = _parse_ljson_v2(lms_dict)
    return LandmarkGroup(pointcloud, labels_to_masks)


def build_model(name, training_dir, group, target_dir,
                n_levels=3, downscale=1.5, crop_percentage=0.25,
                normalization_diagonal=100, max_images=None,
                verbose=False, force=False):
    # Test if model already exists in target location
    model_dir = _model_directory(target_dir, name)
    if p.exists(model_dir):
        if p.isdir(model_dir):
            if not force:
                raise IOError(
                    "{} already exists, remove it or restart with the --force "
                    "flag".format(model_dir))
            else:
                shutil.rmtree(model_dir)
                os.makedirs(model_dir)
        else:
            raise IOError("{} is not a directory".format(model_dir))
    else:
        os.makedirs(model_dir)

    # Load up images
    images = []
    image_gen = mio.import_images(training_dir,
                                  max_images=max_images, verbose=verbose)

    for i in image_gen:
        if i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')
        images.append(i)

    # Create an AAM model
    if verbose:
        print("\nTraining AAM")

    aam = AAMBuilder(
        features=fast_dsift, normalization_diagonal=normalization_diagonal,
        n_levels=n_levels, downscale=downscale, scaled_shape_models=False,
        max_appearance_components=None
    ).build(images, group=group, verbose=verbose)

    # Store pickled versions on disk
    mio.export_pickle(aam, p.join(model_dir, 'aam.pkl'))

    with open(p.join(model_dir, 'info.json'), 'w') as info_f:
        json.dump({
            "n_levels": n_levels, "name": name, "group": group
        }, info_f)

    if verbose:
        print("Created 'aam.pkl', 'info.json' in {}".format(model_dir))

    return aam


def load_model(name, source_dir):
    model_dir = _model_directory(source_dir, name)
    model = mio.import_pickle(p.join(model_dir, 'aam.pkl'))
    with open(p.join(model_dir, 'info.json'), 'r') as info_f:
        info = json.load(info_f)

    fitter = LucasKanadeAAMFitter(model, n_shape=[3, 6, 12], n_appearance=0.3)
    detector = load_dlib_frontal_face_detector()

    return model, info, fitter, detector


def base64_to_image(encoded):
    # Decode base64 url
    clean_encoded = encoded.split('base64,')[1].strip()
    pil_img = PILImage.open(BytesIO(base64.b64decode(clean_encoded)))
    mode = pil_img.mode

    # Copied over from menpo.io.input.image for now as it only handle files
    # and we want to be able to do this from memory, the client always Sends
    # PNG representation of canvas so we can default to the RGBA mode
    if mode == 'RGBA':
        alpha = np.array(pil_img)[..., 3].astype(np.bool)
        image_pixels = _pil_to_numpy(pil_img, True, convert='RGB')
        image = (MaskedImage(image_pixels, mask=alpha)
                 .as_unmasked(copy=False)
                 .as_greyscale(mode='luminosity'))
    else:
        raise ValueError('Unexpected mode for PIL: {}'.format(mode))

    return image


def _pil_to_numpy(pil_img, normalise, convert=None):
    dtype = np.float if normalise else None
    p = pil_img.convert(convert) if convert else pil_img
    np_pixels = np.array(p, dtype=dtype, copy=True)
    if len(np_pixels.shape) is 3:
        np_pixels = np.rollaxis(np_pixels, -1)
    return np_pixels * (1.0 / 255.0) if normalise else np_pixels


class Fitter(object):

    def __init__(self, name, directory, batch_size=50):
        self.model_dir = _model_directory(directory, name)
        aam, info, fitter, detector = load_model(name, directory)
        self.model = aam
        self.info = info
        self.fitter = fitter
        self.detector = detector

        self.n_received = 0
        self.cache_dir = p.join(self.model_dir, CACHE_DIR)
        self.init_cache()

    def init_cache(self):
        if not p.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._images_cache = {}
        self._landmarks_cache = {}

        for child in Path(self.cache_dir).glob('*'):
            self.n_received += 1
            self.load_from_disk(child.stem)

    def load_from_disk(self, uid):
        with open(p.join(self.cache_dir, uid, IMG_FILE), 'r') as f:
            self._images_cache[uid] = cPickle.load(f)
        try:
            with open(p.join(self.cache_dir, uid, LMS_FILE), 'r') as f:
                self._landmarks_cache = json.load(f)
        except IOError:
            self._landmarks_cache[uid] = None

    def store_img(self, uid, img):
        with open(p.join(self.cache_dir, uid, IMG_FILE), 'w') as f:
            cPickle.dump(img, f)

    def store_landmarks(self, uid, lms):
        if (lms):
            with open(p.join(self.cache_dir, uid, LMS_FILE), 'w') as f:
                json.dump(lms, f)

    def receive(self, encoded_img, landmarks=None):
        # Trust client uid if it doesn't exist, they should not
        # interfere with the uuid4 scheme anyways
        uid = str(uuid.uuid4())
        img = base64_to_image(encoded_img)
        os.makedirs(p.join(self.cache_dir, uid))
        self._images_cache[uid] = img
        self.store_img(uid, img)
        self.n_received += 1

        if landmarks:
            self.update_landmarks(uid, landmarks)

        return self.fit(uid)

    def fit(self, uid, landmarks=None):

        if landmarks:
            self.update_landmarks(uid, landmarks)

        # Load image
        img = self._images_cache[uid]

        # Detect the bounding the box, for now assuming this is a face shape
        self.detector(img)

        # Putting the reference shape in the bounding box
        ref_shape = self.fitter.reference_shape

        bbox = img.landmarks.get('dlib_0', None)

        if not bbox:
            return dict(uid=uid, result=None, error="Failed to detect face")

        bbox = bbox.lms
        shape_bb = ref_shape.bounding_box()
        init_shape = AlignmentSimilarity(shape_bb, bbox).apply(ref_shape)

        # Perform fitting
        try:
            result = self.fitter.fit(img, init_shape)
            img.landmarks['fit'] = result.final_shape
            return dict(uid=uid, result=img.landmarks['fit'].tojson())
        except Exception:
            return dict(uid=uid, result=None)

    def update_landmarks(self, uid, landmarks, ground_truth=False):
        ljson = None
        if landmarks:
            self._landmarks_cache[uid] = _parse_ljson(landmarks)
            ljson = self._landmarks_cache[uid].tojson()
            if ground_truth:
                self.store_landmarks(uid, ljson)
        else:
            self._landmarks_cache[uid] = None
        return dict(uid=uid, landmarks=ljson)


class FitAdapter(object):

    """
    Simple adapter class passing requests though a list of fitter identified
    by their name and loaded from one unique directory
    """

    def __init__(self, model_dir):
        self.model_dir = Path(p.abspath(p.expanduser(model_dir)))
        self.fitters = {id: Fitter(id, str(self.model_dir))
                        for id in self.model_ids()}

    def model_ids(self):
        return [m.stem for m in self.model_dir.glob('*')]

    def fit(self, type, uid, landmarks=None):
        return self.fitters[type].fit(uid, landmarks=landmarks)

    def receive(self, type, encoded_img, landmarks=None):
        return self.fitters[type].receive(encoded_img, landmarks=landmarks)

    def update(self, type, uid, landmarks):
            return self.fitters[type].update_landmarks(
                uid, landmarks, ground_truth=True)
