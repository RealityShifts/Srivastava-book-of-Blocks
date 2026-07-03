# Index

## core

Core neural-network primitives.

* Linear
* ConvBlock
* DepthwiseSeparableConv2d
* DilatedConv2d
* GroupConv2d
* Conv1d
* Conv3d
* Mish
* RMSNorm
* AdaIN
* SPADE
* ResidualBlock
* SkipConnection

## attention

Attention mechanisms.

* MultiHeadAttention
* SelfAttention
* CausalSelfAttention
* CrossAttention
* WindowAttention
* LinearAttention
* FlashAttention
* RotaryEmbedding
* RelativePositionBias
* AttentionPooling

## transformer

Transformer encoder / decoder, FFN variants, MoE.

* FeedForward
* SwiGLU
* GEGLU
* TransformerEncoderBlock
* TransformerDecoderBlock
* MixtureOfExperts
* SwitchMoE

## cnn_vision

CNN and vision-specific blocks.

* InceptionBlock
* DenseBlock
* SqueezeExcitation
* CBAM
* SpatialPyramidPooling
* FeaturePyramidNetwork
* ASPP
* PixelShuffleUpsample
* DeformableConv2d
* DeformableAttention

## unet_diffusion

UNet, time conditioning, ControlNet, LoRA, hypernets.

* SinusoidalTimeEmbedding
* TimestepMLP
* DownsampleBlock
* UpsampleBlock
* UNetResBlock
* UNet
* NoisePredictor
* ZeroConv2d
* ControlNetBlock
* LoRALinear
* LoRAConv2d
* HyperNetwork
* IPAdapterCrossAttention

## gan

GAN building blocks: StyleGAN, PGGAN, equalised LR.

* EqualLinear
* EqualConv2d
* GeneratorBlock
* DiscriminatorBlock
* MappingNetwork
* StyleBlock
* ModulatedConv2d
* MinibatchStdDev
* ProgressiveGrowing

## vit

Vision Transformer blocks.

* PatchEmbedding
* CLSToken
* SwinWindowAttention
* ShiftedWindowAttention
* MaskedImageModeling

## sequence

Recurrent and state-space sequence models.

* RNNCell
* LSTMCell
* GRUCell
* StateSpaceModel
* MambaBlock

## gnn

Graph neural network layers.

* MessagePassing
* GraphConv
* GraphAttention

## generative

VAE, autoregressive, normalising-flow, EBM, diffusion schedulers.

* VAE
* MaskedConv2d
* AutoregressiveBlock
* AffineCouplingLayer
* EnergyBasedModel
* DDPMScheduler
* DDIMScheduler

## rl

Reinforcement-learning building blocks.

* PolicyNetwork
* ValueNetwork
* QNetwork
* ActorCritic
* ReplayBuffer
* TargetNetwork

## memory_retrieval

External memory, vector stores, RAG, KV caches.

* ExternalMemory
* VectorStore
* RAGModule
* KVCache

## embedding

Token / positional / projection embeddings, contrastive losses.

* TokenEmbedding
* LearnedPositionalEmbedding
* SinusoidalPositionalEmbedding
* ProjectionHead
* CLIPLoss
* info_nce

## optimization

Optimisers, schedulers, EMA, mixed-precision, checkpointing.

* Lion
* Sophia
* EMA
* MixedPrecisionTrainer
* CheckpointedSequential

## multimodal

Multimodal / agentic blocks.

* CLIPEncoder
* PerceiverResampler
* QFormer
* ToolUseBlock
* MemoryAttention

## efficient

Sparsity, quantisation, parallelism, low-rank.

* QuantizedLinearInt8
* QuantizedLinear4bit
* MagnitudePruner
* TokenPruner
* LowRankLinear
* ColumnParallelLinear
* RowParallelLinear
* PipelineStage

## specialized

Specialised research blocks (NeuralODE, FNO, KAN, capsules, slots).

* NeuralODE
* SpectralConv2d
* FNOBlock
* KANLayer
* CapsuleLayer
* SlotAttention

## tiny_transformer_lm

Minimal causal LM-style stack built from Book-of-Blocks embedding + transformer blocks.

* [TinyTransformerLM](tiny_transformer_lm/TinyTransformerLM.md)
