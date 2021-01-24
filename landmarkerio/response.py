from functools import partial

from sanic import response
from sanic.response import HTTPResponse

from landmarkerio import Mimetype
from landmarkerio.types import PathLike


async def serve_file(mimetype: str, path: PathLike, gzip: bool = False) -> HTTPResponse:
    headers = None
    if gzip:
        headers = {"Content-Encoding": "gzip"}
    return await response.file(path, mime_type=mimetype, headers=headers)


serve_image_file = partial(serve_file, Mimetype.jpeg)
serve_binary_file = partial(serve_file, Mimetype.binary)
serve_gzip_binary_file = partial(serve_binary_file, gzip=True)
