import base64
from io import BytesIO
from PIL import Image as PILImage
import numpy as np
from menpo.image import Image
from menpofit.dlib import DlibWrapper
from menpodetect.dlib import load_dlib_frontal_face_detector


def base64_to_image(encoded):
    # Decode base64 url
    clean_encoded = encoded.split('base64,')[1].strip()
    pil_img = PILImage.open(BytesIO(base64.b64decode(clean_encoded)))
    mode = pil_img.mode

    # Copied over from menpo.io.input.image for now as it only handle files
    # and we want to be able to do this from memory, the client always Sends
    # PNG representation of canvas so we can default to the RGBA mode
    if mode == 'RGBA':
        image_pixels = _pil_to_numpy(pil_img, False, convert='RGB')
        print(image_pixels.dtype)
        image = Image(image_pixels, copy=False)
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


class DLibFitter(object):

    def __init__(self, dlib_predictor_path):
        self.fitter = DlibWrapper(dlib_predictor_path)
        self.detector = load_dlib_frontal_face_detector()

    def __call__(self, encoded_img, landmarks=None):

        # Load image
        img = base64_to_image(encoded_img)

        self.detector(img)
        bbox = img.landmarks.get('dlib_0', None)

        if not bbox:
            return {
              "error": "Failed to detect face"
            }

        # Perform fitting
        try:
            fr = self.fitter.fit_from_bb(img, bbox.lms)
            img.landmarks['fit'] = fr.final_shape
            return {
                'result': img.landmarks['fit'].tojson(),
            }
        except Exception as e:
            return {
                'result': None,
                'error': str(e)
            }


class FitAdapter(object):

    """
    Simple adapter class passing requests though a list of fitter identified
    by their name and loaded from one unique directory
    """

    def __init__(self, dlib_predictor_path):
        self.fitters = {
            'ibug68': DLibFitter(dlib_predictor_path)
        }

    def model_ids(self):
        return self.fitters.keys()

    def fit(self, group, encoded_img, landmarks=None):
        return self.fitters[group](encoded_img, landmarks=landmarks)
