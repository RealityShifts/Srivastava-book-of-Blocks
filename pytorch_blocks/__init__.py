"""PyTorch implementations of famous ML/CV/NLP building blocks.

The package is organized into one file per architectural family:

    core_blocks            - Linear / Conv / Norm / Activation / Residual
    attention_blocks       - Self / Cross / Causal / Sparse / Linear / RoPE / ...
    transformer_blocks     - Encoder / Decoder / FFN variants / MoE
    cnn_vision_blocks      - Inception / Dense / SE / CBAM / FPN / ASPP / ...
    unet_diffusion_blocks  - UNet / Time embed / CFG / ControlNet / LoRA / IP-Adapter
    gan_blocks             - Generator / Discriminator / AdaIN / ModulatedConv / ...
    vit_blocks             - PatchEmbed / CLS / Window / ShiftedWindow / MAE
    sequence_blocks        - RNN / LSTM / GRU / SSM / SelectiveScan
    gnn_blocks             - MessagePassing / GraphConv / GAT
    generative_blocks      - VAE / Reparam / AR / Flow / EBM / Schedulers
    rl_blocks              - Policy / Value / ActorCritic / Replay / Target net
    memory_retrieval_blocks- ExternalMemory / RAG / KVCache
    embedding_blocks       - Token / Positional / Contrastive / ProjectionHead
    optimization_blocks    - Lion / Sophia / Schedulers / EMA / Mixed-precision
    multimodal_blocks      - CLIP / PerceiverResampler / Q-Former / Tool / MemoryAttn
    efficient_blocks       - Quantization / Pruning / LowRank / Tensor & Pipeline parallel
    specialized_blocks     - Neural ODE / FNO / KAN / Capsule / SlotAttention

Only dependency: ``torch``.
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
    filters,
)

__all__ = [
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
    "filters",
]
