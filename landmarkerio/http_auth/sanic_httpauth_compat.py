# Borrowed under the terms of the MIT license
# https://github.com/MihaiBalint/Sanic-HTTPAuth/blob/master/LICENSE
# Borrowed code from werkzeug: https://github.com/pallets/werkzeug
import base64
import hmac
import logging
import sanic.response
import sys

from sanic.request import Request
from urllib.request import parse_http_list as _parse_list_header

log = logging.getLogger(__name__)

_builtin_safe_str_cmp = getattr(hmac, "compare_digest", None)


def safe_str_cmp(a, b):
    """This function compares strings in somewhat constant time.  This
    requires that the length of at least one string is known in advance.
    Returns `True` if the two strings are equal, or `False` if they are not.
    .. versionadded:: 0.7
    """
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")

    if _builtin_safe_str_cmp is not None:
        return _builtin_safe_str_cmp(a, b)

    if len(a) != len(b):
        return False

    rv = 0
    if PY2:
        for x, y in izip(a, b):
            rv |= ord(x) ^ ord(y)
    else:
        for x, y in izip(a, b):
            rv |= x ^ y

    return rv == 0


class ImmutableDictMixin(object):
    """Makes a :class:`dict` immutable.
    .. versionadded:: 0.5
    :private:
    """

    _hash_cache = None

    @classmethod
    def fromkeys(cls, keys, value=None):
        instance = super(cls, cls).__new__(cls)
        instance.__init__(zip(keys, repeat(value)))
        return instance

    def __reduce_ex__(self, protocol):
        return type(self), (dict(self),)

    def _iter_hashitems(self):
        return iteritems(self)

    def __hash__(self):
        if self._hash_cache is not None:
            return self._hash_cache
        rv = self._hash_cache = hash(frozenset(self._iter_hashitems()))
        return rv

    def setdefault(self, key, default=None):
        is_immutable(self)

    def update(self, *args, **kwargs):
        is_immutable(self)

    def pop(self, key, default=None):
        is_immutable(self)

    def popitem(self):
        is_immutable(self)

    def __setitem__(self, key, value):
        is_immutable(self)

    def __delitem__(self, key):
        is_immutable(self)

    def clear(self):
        is_immutable(self)


class Authorization(ImmutableDictMixin, dict):
    """Represents an `Authorization` header sent by the client.  You should
    not create this kind of object yourself but use it when it's returned by
    the `parse_authorization_header` function.
    This object is a dict subclass and can be altered by setting dict items
    but it should be considered immutable as it's returned by the client and
    not meant for modifications.
    .. versionchanged:: 0.5
       This object became immutable.
    """

    def __init__(self, auth_type, data=None):
        dict.__init__(self, data or {})
        self.type = auth_type

    username = property(
        lambda self: self.get("username"),
        doc="""
        The username transmitted.  This is set for both basic and digest
        auth all the time.""",
    )
    password = property(
        lambda self: self.get("password"),
        doc="""
        When the authentication type is basic this is the password
        transmitted by the client, else `None`.""",
    )
    realm = property(
        lambda self: self.get("realm"),
        doc="""
        This is the server realm sent back for HTTP digest auth.""",
    )
    nonce = property(
        lambda self: self.get("nonce"),
        doc="""
        The nonce the server sent for digest auth, sent back by the client.
        A nonce should be unique for every 401 response for HTTP digest
        auth.""",
    )
    uri = property(
        lambda self: self.get("uri"),
        doc="""
        The URI from Request-URI of the Request-Line; duplicated because
        proxies are allowed to change the Request-Line in transit.  HTTP
        digest auth only.""",
    )
    nc = property(
        lambda self: self.get("nc"),
        doc="""
        The nonce count value transmitted by clients if a qop-header is
        also transmitted.  HTTP digest auth only.""",
    )
    cnonce = property(
        lambda self: self.get("cnonce"),
        doc="""
        If the server sent a qop-header in the ``WWW-Authenticate``
        header, the client has to provide this value for HTTP digest auth.
        See the RFC for more details.""",
    )
    response = property(
        lambda self: self.get("response"),
        doc="""
        A string of 32 hex digits computed as defined in RFC 2617, which
        proves that the user knows a password.  Digest auth only.""",
    )
    opaque = property(
        lambda self: self.get("opaque"),
        doc="""
        The opaque header from the server returned unchanged by the client.
        It is recommended that this string be base64 or hexadecimal data.
        Digest auth only.""",
    )
    qop = property(
        lambda self: self.get("qop"),
        doc="""
        Indicates what "quality of protection" the client has applied to
        the message for HTTP digest auth. Note that this is a single token,
        not a quoted list of alternatives as in WWW-Authenticate.""",
    )


def to_unicode(
    x, charset=sys.getdefaultencoding(), errors="strict", allow_none_charset=False
):
    if x is None:
        return None
    if not isinstance(x, bytes):
        return str(x)
    if charset is None and allow_none_charset:
        return x
    return x.decode(charset, errors)


def bytes_to_wsgi(data):
    assert isinstance(data, bytes), "data must be bytes"
    if isinstance(data, str):
        return data
    else:
        return data.decode("latin1")


def wsgi_to_bytes(data):
    """coerce wsgi unicode represented bytes to real ones"""
    if isinstance(data, bytes):
        return data
    return data.encode("latin1")  # XXX: utf8 fallback?


def unquote_header_value(value, is_filename=False):
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.
    .. versionadded:: 0.5
    :param value: the header value to unquote.
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != "\\\\":
            return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def parse_dict_header(value, cls=dict):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict (or any other mapping object created from
    the type with a dict like interface provided by the `cls` argument):
    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]
    If there is no value for a key it will be `None`:
    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}
    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.
    .. versionchanged:: 0.9
       Added support for `cls` argument.
    :param value: a string with a dict header.
    :param cls: callable to use for storage of parsed results.
    :return: an instance of `cls`
    """
    result = cls()
    if not isinstance(value, str):
        # XXX: validate
        value = bytes_to_wsgi(value)
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
            continue
        name, value = item.split("=", 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result


def parse_authorization_header(value):
    """Parse an HTTP basic/digest authorization header transmitted by the web
    browser. The return value is either `None` if the header was invalid or
    not given, otherwise an :class:`~werkzeug.datastructures.Authorization`
    object.
    :param value: the authorization header to parse.
    :return: a :class:`~werkzeug.datastructures.Authorization` object or `None`.
    """
    if not value:
        return
    value = wsgi_to_bytes(value)
    try:
        auth_type, auth_info = value.split(None, 1)
        auth_type = auth_type.lower()
    except ValueError:
        return
    if auth_type == b"basic":
        try:
            username, password = base64.b64decode(auth_info).split(b":", 1)
        except Exception:
            return
        return Authorization(
            "basic",
            {
                "username": to_unicode(username, "utf-8"),
                "password": to_unicode(password, "utf-8"),
            },
        )
    elif auth_type == b"digest":
        auth_map = parse_dict_header(auth_info)
        for key in "username", "realm", "nonce", "uri", "response":
            if key not in auth_map:
                return
        if "qop" in auth_map:
            if not auth_map.get("nc") or not auth_map.get("cnonce"):
                return
        return Authorization("digest", auth_map)


def get_request(*args, **kwargs):
    for a in args:
        if isinstance(a, Request):
            return a
    for k, v in kwargs.items():
        if isinstance(v, Request):
            return v
