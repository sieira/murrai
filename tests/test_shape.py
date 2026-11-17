import dataclasses

import pytest

from murrai.domain.shape import Shape


def test_a_shape_is_frozen():
    shape = Shape(depth=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        shape.depth = 8


def test_a_shape_rejects_contradictory_knobs():
    shape = Shape(depth=4, dimension=64, max_time=64, heads=1)
    with pytest.raises(ValueError, match="a checkpoint keeps its own shape"):
        shape.reject_contradictory(depth=8)


def test_a_shape_accepts_consistent_knobs():
    shape = Shape(depth=4, dimension=64, max_time=64, heads=1)
    shape.reject_contradictory(depth=4, dimension=64, max_time=64, heads=1)


def test_a_shape_ignores_unspecified_knobs():
    shape = Shape(depth=4)
    shape.reject_contradictory(dimension=None, max_time=None)


def test_a_shape_ignores_none_for_unset_fields():
    shape = Shape(depth=4)
    shape.reject_contradictory(window=4, hidden=128)
