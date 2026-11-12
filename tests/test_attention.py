import math

import pytest
import torch

from murrai.domain.attention import CausalSelfAttention


@pytest.fixture
def attention():
    torch.manual_seed(0)
    return CausalSelfAttention(8)


def test_output_shape_matches_input_shape(attention):
    x = torch.randn(2, 5, 8)
    assert attention(x).shape == x.shape


def test_perturbing_a_position_leaves_earlier_answers_bit_identical(attention):
    torch.manual_seed(1)
    x = torch.randn(1, 6, 8)
    before = attention(x)
    perturbed = x.clone()
    perturbed[:, 3] += torch.randn(1, 8) * 10
    after = attention(perturbed)
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3:], after[:, 3:])


def test_every_query_row_shares_its_attention_across_seen_positions_only(
    attention,
):
    x = torch.randn(1, 4, 8)
    attention(x)
    shares = attention.shares
    assert shares.shape == (1, 1, 4, 4)
    sums = shares.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_a_matching_position_draws_more_attention_than_a_stranger(attention):
    d = 8
    identity = torch.eye(d)
    attention.query.weight.data.copy_(identity)
    attention.key.weight.data.copy_(identity)
    attention.value.weight.data.copy_(identity)
    chorus = torch.randn(1, 1, d) * 3
    stranger = -chorus
    row = torch.cat([chorus, stranger, chorus], dim=1)
    attention(row)
    shares = attention.shares[0, 0, 2]
    assert shares[0] > shares[1]


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_output_shape_holds_for_every_head_count(heads):
    torch.manual_seed(0)
    attention = CausalSelfAttention(8, heads=heads)
    x = torch.randn(3, 5, 8)
    assert attention(x).shape == (3, 5, 8)
    assert attention.shares.shape == (3, heads, 5, 5)


def test_perturbing_a_position_leaves_earlier_answers_identical_with_four_heads():
    torch.manual_seed(0)
    attention = CausalSelfAttention(8, heads=4)
    torch.manual_seed(1)
    x = torch.randn(1, 6, 8)
    before = attention(x)
    perturbed = x.clone()
    perturbed[:, 3] += torch.randn(1, 8) * 10
    after = attention(perturbed)
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3:], after[:, 3:])


def single_head_by_hand(attention, x, scale):
    q, k, v = attention.query(x), attention.key(x), attention.value(x)
    scores = q @ k.transpose(-2, -1) / scale
    time = x.size(-2)
    future = torch.triu(torch.ones(time, time, dtype=torch.bool), diagonal=1)
    shares = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    return attention.out(shares @ v)


def test_one_head_equals_the_hand_built_single_head_answer():
    torch.manual_seed(0)
    attention = CausalSelfAttention(8, heads=1)
    x = torch.randn(2, 5, 8)
    expected = single_head_by_hand(attention, x, math.sqrt(8))
    torch.testing.assert_close(attention(x), expected)


def test_each_head_scales_scores_by_its_own_slice_width():
    torch.manual_seed(0)
    attention = CausalSelfAttention(8, heads=4)
    x = torch.randn(1, 5, 8)
    attention(x)
    q = attention.query(x)[0, :, 2:4]
    k = attention.key(x)[0, :, 2:4]
    scores = q @ k.T / math.sqrt(2)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    expected = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    torch.testing.assert_close(attention.shares[0, 1], expected)


def test_changing_one_slice_of_query_and_key_changes_only_that_head():
    torch.manual_seed(0)
    attention = CausalSelfAttention(8, heads=2)
    x = torch.randn(1, 5, 8)
    attention(x)
    before = attention.shares.clone()
    torch.manual_seed(5)
    attention.query.weight.data[4:] = torch.randn(4, 8)
    attention.key.weight.data[4:] = torch.randn(4, 8)
    attention(x)
    after = attention.shares
    assert torch.equal(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_a_dimension_the_heads_do_not_divide_is_refused():
    with pytest.raises(ValueError):
        CausalSelfAttention(10, heads=4)


def test_the_head_count_does_not_change_the_parameter_count():
    counts = {
        sum(p.numel() for p in CausalSelfAttention(8, heads=heads).parameters())
        for heads in (1, 2, 4)
    }
    assert counts == {4 * 8 * 8}
