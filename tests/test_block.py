import pytest
import torch

from murrai.domain.block import TransformerBlock


@pytest.fixture
def block():
    torch.manual_seed(0)
    return TransformerBlock(8)


def test_output_shape_matches_input_shape(block):
    x = torch.randn(2, 5, 8)
    assert block(x).shape == x.shape


def test_the_block_acts_on_its_input(block):
    x = torch.randn(1, 6, 8)
    assert not torch.equal(block(x), x)


def test_zeroing_the_edit_path_returns_the_input_unchanged(block):
    x = torch.randn(1, 6, 8)
    block.attend.out.weight.data.zero_()
    block.mlp_down.weight.data.zero_()
    block.mlp_down.bias.data.zero_()
    assert torch.equal(block(x), x)


def test_the_block_passes_its_head_count_to_attention():
    block = TransformerBlock(8, heads=4)
    assert block.attend.heads == 4
    assert block(torch.randn(2, 5, 8)).shape == (2, 5, 8)
