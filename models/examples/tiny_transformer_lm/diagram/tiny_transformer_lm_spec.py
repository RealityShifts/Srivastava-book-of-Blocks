"""Architecture spec for the tiny_transformer_lm example.

Generate:
  python ../../../diagram/_generate.py \\
    --specs tiny_transformer_lm_spec.py \\
    --out . --no-builtins --depth 2
"""

from _generate import _io, _op, _ref, _norm

CATEGORY = "tiny_transformer_lm"
CATEGORY_DESC = (
    "Minimal causal LM-style stack built from Book-of-Blocks "
    "embedding + transformer blocks."
)

BLOCKS = {
    "TinyTransformerLM": (
        "Token + learned position encodings, two pre-norm encoder blocks, "
        "final LayerNorm, tied vocabulary projection.",
        "ids:(B, T) → logits:(B, T, V)",
        [
            [_io("ids  (B, T)")],
            [_ref("TokenEmbedding")],
            [_ref("LearnedPositionalEmbedding")],
            [_ref("TransformerEncoderBlock")],
            [_ref("TransformerEncoderBlock")],
            [_norm("LayerNorm")],
            [_op("tied linear  h @ Eᵀ  (D → V)")],
            [_io("logits  (B, T, V)")],
        ],
    ),
}