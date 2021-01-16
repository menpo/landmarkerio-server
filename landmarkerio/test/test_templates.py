from pathlib import Path

import menpo

from landmarkerio.template import load_template

TEST_DIR = Path(__file__).parent

IBUG68_TEMPLATE_PATH = TEST_DIR / "../default_templates/ibug68.yml"


def test_parse_ibug68_template():
    template = load_template(IBUG68_TEMPLATE_PATH, 2)
    assert template["version"] == 2
    # Lazy test by just parsing with menpo
    menpo.io.input.landmark._parse_ljson_v2(template)
