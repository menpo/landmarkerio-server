import hashlib
import os
from typing import Dict

from loguru import logger

DEFAULT_SALT = "DEFAULT_LANDMARKERIO_SALT"
APP_SALT = os.getenv("LANDMARKERIO_SALT", DEFAULT_SALT)


# Create default sentinel that cannot match with anything
class _MISSING:
    pass


MISSING = _MISSING()


def hash_password(salt: str, password: str) -> str:
    salted = password + salt
    return hashlib.sha512(salted.encode("utf8")).hexdigest()


def validate_salt(is_dev: bool = False) -> None:
    if APP_SALT == DEFAULT_SALT:
        logger.warning("Change default salt before deploying to production")
        if is_dev:
            raise ValueError(
                "Cannot start server without changing default salt by setting "
                "environment variable 'LANDMARKERIO_SALT' or running in dev mode"
            )


def verify_password(users: Dict[str, str], username: str, password: str) -> bool:
    return users.get(username, MISSING) == hash_password(APP_SALT, password)
