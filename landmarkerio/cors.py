r"""
This is the version of the crossdomain wrapper that I have submitted back to
the flask-restful project which adds the ability to set the credentials flag.

    https://github.com/twilio/flask-restful/pull/276

Until it gets merged I'll maintain it here.

"""


from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(allowed_origins=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True, credentials=False):
    """
    http://flask.pocoo.org/snippets/56/
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if isinstance(allowed_origins, str):
        # always have allowed_origins as a list of strings.
        allowed_origins = [allowed_origins]
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            # Get a hold of the request origin
            origin = request.environ.get('HTTP_ORIGIN')
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            # if the origin matches any of our allowed origins set the
            # access control header appropriately
            allow_origin = (origin if origin is not None and
                                      allowed_origins is not None and
                                      origin in allowed_origins else None)
            h['Access-Control-Allow-Origin'] = allow_origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if credentials:
                h['Access-Control-Allow-Credentials'] = 'true'
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator
