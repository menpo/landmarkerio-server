from functools import partial

from sanic import response
from sanic.exceptions import abort

from landmarkerio import Mimetype

url = lambda *x: "/" + "/".join(x)
asset = lambda f: partial(f, "<string:asset_id>")


def safe_send(x, fail_message):
    try:
        return x
    except Exception as e:
        print(e)
        return abort(404, message=fail_message)


async def safe_send_file(mimetype, path, fail_message, gzip=False):
    try:
        return await response.file(path, mime_type=mimetype)
    except Exception as e:
        print(e)
        return abort(404, message=fail_message)


image_file = partial(safe_send_file, Mimetype.jpeg)
binary_file = partial(safe_send_file, Mimetype.binary)
gzip_binary_file = partial(binary_file, gzip=True)
