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
    assert shares.shape == (1, 4, 4)
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
    shares = attention.shares[0, 2]
    assert shares[0] > shares[1]
