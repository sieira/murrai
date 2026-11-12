import math

import pytest

from murrai.domain.autograd import Value, finite_difference_grads


def test_mul_backward_gives_each_factor_the_others_slope():
    a = Value(3.0)
    b = Value(2.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(2.0)
    assert b.grad == pytest.approx(3.0)
    assert c.grad == pytest.approx(1.0)


def test_add_backward_passes_the_slope_unchanged_to_both_parents():
    a = Value(3.0)
    b = Value(4.0)
    c = a + b
    c.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)


def test_reused_value_sums_gradients_from_every_path():
    a = Value(2.0)
    c = a * a + a
    c.backward()
    assert c.grad == pytest.approx(1.0)
    assert a.grad == pytest.approx(5.0)


def test_a_value_survives_a_trip_to_another_process():
    import pickle

    value = Value(2.5)
    revived = pickle.loads(pickle.dumps(value))
    assert revived.data == 2.5


def test_subtract_passes_reversed_sign_to_the_right_parent():
    a = Value(10.0)
    b = Value(4.0)
    c = a - b
    c.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(-1.0)


def test_division_distributes_slope_as_quotient_rule():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert a.grad == pytest.approx(1 / 2)
    assert b.grad == pytest.approx(-6.0 / 4)


def test_power_pumps_the_slope_by_the_exponent():
    a = Value(3.0)
    c = a**2
    c.backward()
    assert a.grad == pytest.approx(6.0)


def test_tanh_squash_rescales_the_slope():
    a = Value(0.5)
    c = a.tanh()
    c.backward()
    assert c.data == pytest.approx(math.tanh(0.5))
    assert a.grad == pytest.approx(1 - math.tanh(0.5) ** 2)


def test_exp_local_slope_is_the_output_itself():
    a = Value(1.0)
    c = a.exp()
    c.backward()
    assert a.grad == pytest.approx(math.exp(1.0))


def test_log_local_slope_is_one_over_the_input():
    a = Value(4.0)
    c = a.log()
    c.backward()
    assert a.grad == pytest.approx(0.25)


def test_operators_accept_plain_numbers_in_either_position():
    a = Value(2.0)
    assert (a + 1).data == pytest.approx(3.0)
    assert (1 + a).data == pytest.approx(3.0)
    assert (a - 1).data == pytest.approx(1.0)
    assert (1 - a).data == pytest.approx(-1.0)
    assert (a * 2).data == pytest.approx(4.0)
    assert (2 * a).data == pytest.approx(4.0)
    assert (a / 2).data == pytest.approx(1.0)
    assert (2 / a).data == pytest.approx(1.0)


def test_zero_grad_resets_the_handle():
    a = Value(2.0)
    out = a * a + a
    out.backward()
    a.zero_grad()
    assert a.grad == 0.0


def test_backward_matches_finite_differences_on_seeded_graph():
    a, b, c = Value(1.5), Value(2.0), Value(0.5)

    def build(a, b, c):
        return (a * b + c).tanh() + a.log() * b + c / (b.exp() + 1)

    out = build(a, b, c)
    out.backward()
    nudged = finite_difference_grads(build, [a, b, c])
    for handle, numeric in zip((a.grad, b.grad, c.grad), nudged, strict=False):
        assert handle == pytest.approx(numeric, rel=1e-4)


def test_gradients_agree_with_number_for_number_with_torch():
    import torch

    a, b, c = Value(1.5), Value(2.0), Value(0.5)

    def build(a, b, c):
        return (a * b + c).tanh() + a.log() * b + c / (b.exp() + 1)

    build(a, b, c).backward()

    tensors = tuple(
        torch.tensor(x.data, requires_grad=True, dtype=torch.float64) for x in (a, b, c)
    )
    build(*tensors).backward()
    for handle, tensor in zip((a.grad, b.grad, c.grad), tensors, strict=False):
        assert handle == pytest.approx(tensor.grad.item(), rel=1e-9)
