"""Tests of the full transformer."""

from annotated.transformer import make_model


def test_transformer() -> None:
    """Test making the model."""
    tmp_model = make_model(src_vocab_size=10, tgt_vocab_size=10, num_layers=2)
    print("Behold The Transformer!")
    print()
    print(tmp_model)
    print()
