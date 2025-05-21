# Genie World Model

# ST Transformer

memory efficient architecture across all model components balancing capability with constraints

t spatiotemporal blocks with interleaved spatial/temporal attention layers followed by feedforward layer as standard attention blocks

Self attention in spatial layer attends over 1 x H x W tokens within each timestep, and in temporal layer attends over T x 1 x 1 across T timesteps

FFW after both spatial and temporal components, and no post-spatial FFW to allow scaling up other model components

# Architecture 

Latent Action Model: Infer the action between two frames

VQVAE video tokenizer: Tokenize video into discrete tokens

Dynamics Model: Given latent action and past frame tokens, predict latent next frame of video tokens

# Latent Action Model

Learn latent actions in unsupervised manner so we can condition dynamics on action without needing action labels for data

Encoder takes in all previous frames $(x_1...x_t)$ and next frame $x_{t + 1}$, outputs corresponding set of continuous latent actions $(a_1...a_t)$

Decoder takes in all previous frames and latent actions as input and predicts the next frame $x_{t + 1}$

To train model, use VQ-VAE objective to limit number of predicted actions to small discrete set of codes

Limit vocab size $|A|$ of VQ codebook (max number of possible latent actions)

Decoder only has access to history and latent action, $a_t$ should encode most meaningful change between past and future for decoder to successfully reconstruct future frame

Decoder gives training signal but is abandoned at inference time and replaced with user action

ST-transformer architecture for latent action model with causal mask in temporal layer to take entire video as input and generate actions between frames.

## Video Tokenizer

Compress videos into discrete tokens to reduce dimensionality and have higher quality video generation (use VQVAE)

# Dynamics Model

At timestep $t \in [1, T]$, take in tokenized video and latent action sequences up to $t - 1$ an dpredict next frame tokens $z_t$.

Use ST Transformer to maximize use of causal structure

Cross entropy loss between predicted and gt video tokens

Ranomly mask input tokens at train time according to bernoulli distribution with masking rate sampled uniformly from 0.5 to 1

Common world model practice: concatenate action at time t to corresponding frame

Treating latent actions as additive embeddings for both latent action and dynamics models helps improve controlability of generation

# Inference Time

Player prompts model with initial frame, image tokenized with video encoder, then player specifies discrete latent action to take by choosing integer value in $[0, |A|]$, use to predict next frame

Repeat process autoregressively as actions passed to model and tokens decoded into video frames with decoder

# Train Info

55M 16s video clips at 10FPS with 160 x 90 resolution
Final dataset 6.8M 16s video clips

11B parameter model

200M video tokenizer, patch size of 4, codebook with embedding size 32 and 1024 unique codes

LAM 300M, patch size 16, codebook embedding size 32, 8 unqiue codes (actions)

# Metrics of Success

Frechet Video Distance for video fidelity and Peak Signal-to-Noise Ratio for controllability (how much video generations differ when conditioned on latent actions inferred from ground truth vs sapled from random distribution)

