"""Flax NNX implementations of famous ML / CV / NLP building blocks.

Mirrors the file layout of ``pytorch_blocks/`` so the same architectural
families are kept together:

    core_blocks            - Linear / Conv / Norm / Activation / Residual
    attention_blocks       - Self / Cross / Causal / Sparse / Linear / RoPE
    transformer_blocks     - Encoder / Decoder / FFN variants / MoE
    cnn_vision_blocks      - Inception / Dense / SE / CBAM / FPN / ASPP
    unet_diffusion_blocks  - UNet / Time embed / CFG / ControlNet / LoRA
    gan_blocks             - Generator / Discriminator / AdaIN / ModulatedConv
    vit_blocks             - PatchEmbed / CLS / Window / ShiftedWindow / MAE
    sequence_blocks        - RNN / LSTM / GRU / SSM / SelectiveScan
    gnn_blocks             - MessagePassing / GraphConv / GAT
    generative_blocks      - VAE / Reparam / AR / Flow / EBM / DDPM / DDIM
    rl_blocks              - Policy / Value / ActorCritic / Replay / Target
    memory_retrieval_blocks- ExternalMemory / RAG / KVCache
    embedding_blocks       - Token / Positional / Contrastive / ProjectionHead
    optimization_blocks    - SGD / Adam / Lion / Sophia / Schedulers / EMA
    multimodal_blocks      - CLIP / PerceiverResampler / Q-Former / Tool / MemAttn
    efficient_blocks       - Quantization / Pruning / LowRank / Tensor parallel
    specialized_blocks     - Neural ODE / FNO / KAN / Capsule / SlotAttention

Convention notes (different from the PyTorch port):
    - Tensors are channel-last (NHWC for images, BTC for sequences).
    - Modules accept a keyword-only ``rngs: nnx.Rngs`` for parameter init.
    - Modules whose forward path needs randomness accept an explicit
      ``key: jax.Array`` argument.

Only direct dependencies: ``flax`` and ``jax``.
"""

from . import (
    core_blocks,
    attention_blocks,
    transformer_blocks,
    cnn_vision_blocks,
    unet_diffusion_blocks,
    gan_blocks,
    vit_blocks,
    sequence_blocks,
    gnn_blocks,
    generative_blocks,
    rl_blocks,
    memory_retrieval_blocks,
    embedding_blocks,
    optimization_blocks,
    multimodal_blocks,
    efficient_blocks,
    specialized_blocks,
)
from .core_blocks import Activation, Norm, SkipMode

__all__ = [
    "Activation",
    "Norm",
    "SkipMode",
    "core_blocks",
    "attention_blocks",
    "transformer_blocks",
    "cnn_vision_blocks",
    "unet_diffusion_blocks",
    "gan_blocks",
    "vit_blocks",
    "sequence_blocks",
    "gnn_blocks",
    "generative_blocks",
    "rl_blocks",
    "memory_retrieval_blocks",
    "embedding_blocks",
    "optimization_blocks",
    "multimodal_blocks",
    "efficient_blocks",
    "specialized_blocks",
]
