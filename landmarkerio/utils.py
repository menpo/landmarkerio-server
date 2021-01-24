import os
import os.path as p
from pathlib import Path
from typing import Tuple

from landmarkerio.types import PathLike


def parse_username_and_password_file(path: PathLike) -> Tuple[str, str]:
    path = Path(p.abspath(p.expanduser(path)))
    with path.open("rt") as f:
        user_pass = [line.strip() for line in f.readlines()]
    return user_pass[0], user_pass[1]


DIMS = {"image": 2, "mesh": 3}
