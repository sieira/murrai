import importlib

import pytest

LAYERS = [
    "murrai.composition",
    "murrai.presentation",
    "murrai.adapters",
    "murrai.application",
    "murrai.domain",
]


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_package_exists(layer):
    assert importlib.import_module(layer) is not None
