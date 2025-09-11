# Nano World Model

Inspired by Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT), and Google's Genie 1 Paper (https://arxiv.org/pdf/2402.15391).

The greatest challenge in training world models as opposed to video models is the requirement of action annotations at each timestep. WHen we require actions, we can no longer train on the entire internet's video data and get amazing results like Google's VEO3 (or, hint, Genie 3). 

Genie solves this problem by inferring the actions between frames, and labelling them in an unsupervised manner. This is the critical unlock to achieving scale with world models.

This codebase is meant to help people understand world modeling and a common architecture for doing so (likely semi-similar to Genie 3).

# Architecture 

This world model is autoregressive over discrete tokens, similar to LLMs. We can thus use many of the innovations from LLMs to improve our world model.

FSQ-VAE video tokenizer: Tokenize video into discrete tokens

Latent Action Model: Infer the action between two frames

Dynamics Model: Given latent action and past frame tokens, predict latent next frame of video tokens

# Latent Action Model

Learn latent actions in unsupervised manner so we can condition dynamics on action without needing action labels for data

Encoder takes in all previous frames $(x_1...x_t)$ and next frame $x_{t + 1}$, outputs corresponding set of continuous latent actions $(a_1...a_t)$

Decoder takes in all previous frames and latent actions as input and predicts the next frame $x_{t + 1}$

To train model, use FSQ objective to limit number of predicted actions to small discrete set of codes

Limit vocab size $|A|$ of FSQ codebook (max number of possible latent actions)

Decoder only has access to history and latent action, $a_t$ should encode most meaningful change between past and future for decoder to successfully reconstruct future frame

Decoder gives training signal but is abandoned at inference time and replaced with user action

ST-transformer architecture for latent action model with causal mask in temporal layer to take entire video as input and generate actions between frames.

## Video Tokenizer

Compress videos into discrete tokens to reduce dimensionality and have higher quality video generation (use VQVAE)

# Dynamics Model

At timestep $t \in [1, T]$, take in tokenized video and latent action sequences up to $t - 1$ and predict next frame tokens $z_t$.

Use ST Transformer to maximize use of causal structure

Condition on latent action sequences with Adaptive Layer Norm: conditioning is projected to gamma and beta, then we replace layernorm/RMSNorm in the transformer with layernorm(x) * gamma + (1 + beta)

Cross entropy loss between predicted and gt video tokens

Randomly mask input tokens at train time according to bernoulli distribution with masking rate sampled uniformly from 0.5 to 1

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

Frechet Video Distance for video fidelity and Peak Signal-to-Noise Ratio for controllability (how much video generations differ when conditioned on latent actions inferred from ground truth vs sampled from random distribution)

# ST Transformer

L spatial/temporal blocks with (spatial layer, temporal layer, feedforward layer)
spatial layer attends over 1 x H x W tokens within each timestep
temporal layer attends over T x 1 x 1 across T timesteps for the same token

(so just attention translated to video domain)

memory efficient architecture across all model components balancing capability with constraints

t spatiotemporal blocks with interleaved spatial/temporal attention layers followed by feedforward layer as standard attention blocks

Self attention in spatial layer attends over 1 x H x W tokens within each timestep, and in temporal layer attends over T x 1 x 1 across T timesteps

FFW after both spatial and temporal components, and no post-spatial FFW to allow scaling up other model components

# Finite Scalar Quantization Variational Autoencoder (FSQ-VAE)

Paper: https://arxiv.org/pdf/2309.15505

TLDR, in FSQ:
1. Encoder-decoder use discrete representations/codes
3. We recieve direct gradients for the discretization and require no auxiliary losses, unlike in VQVAE

## Introduction

VQVAE paper argues using discrete latents is potentially more natural, clearly for language, but they argue even for images, which can be described by language.

Discrete latents are appealing since "powerful autoregressive models" (ex: GPTs) have already successful modeled distributions over discrete variables

## VAEs
Consist of:
1. Encoder network to parameterize posterior distribution $q(z | x)$ of discrete latent random variables z fiven input data x
2. Prior distribution $p(z)$
3. Decoder network to parameterize likelihood $p(x | z)$ over input data x given latent z

Notes:
1. Posteriors and priors assumed to be normal distribution and diagonal covariance matrix (each dimension is independent): enables gaussian reparametrization trick
2. Extensions: autoregressive prior/posterior models, normalising flows, inverse autoregressive posteriors

## Finite Scalar Quantization

Posterior and prior distributions are categorical, and samples drawn from these fall into a specific region of a hypercube, which are used as discrete inputs to decoder network.

Latent Embedding Space $|e| = {L^{D}}$ where $L$ is the number of levels/bins per dimension and $D$ is the dimensionality of the hypercube (number of dimensions in the latent embeddings). Essentially, for each dimension, we can fall into one of L buckets, and we have D dimensions. if we had 3 dimensions and 2 levels per dimension, we'd have 2^8 possible regions in the cube.

Input $x$ passed through encoder to produce $z_e(x)$, and discrete latent $z$ is calculated by:

1. tanhing to [-1,1]
2. transforming to [0, L]
3. rounding to the nearest integer
4. transforming back to [-1,1]

the resulting discrete latent vector $e_k$ is used as input to decoder network.
 

-------- TODO: CLARIFY THIS SECTION --------

VQVAE is a VAE where we can bound $log p(x)$ with ELBO
Proposal distribution $q(z = k | x)$ is deterministic and with uniform prior over $z$, so $\sum_z q(z | x) \log \frac{q(z | x)}{p(z)} = \log \frac{1}{1 / K}$ since all other zs have 0 probability besides the deterministic z mapped to by the encoder, and p is uniform over K (so each $p(z)$ is 1/K), and the entropy of q is 0, thus we have KL divergence constant and equal to $0 - \log (1 / K) = \log K$

--------------------------------

## FSQVAE Training

In FSQ, the discretization process has a straight-through gradient, where the input to the decoder is equal to z + stopgrad(z_q - z). So the quantity is z_q, but the gradient is on z.

We thus need no extra loss terms unlike in VQVAE, which makes this method siginificantly more wieldy.

stopgrad is the stopgradient operator (in pytorch, .detach()) that is identity at forward computation time and has zero partial derivatives, and thus constrains the operand to be non-updated constant.


## FSQVAE for Video

We can model 128 x 128 x 3 images by compressing to 32 x 32 x 1 discrete space. We use L = 5 and D = 4 yielding a 1024-length discrete codebook.

Initial 4 frames input to model.

Generation is purely in latent space $z_t$ without need to generate actual imgaes themselves

Each image in sequence $x_t$ created by mapping latents with deterministic decoder to pixel space after all latents generated using prior model $p(z_1,...,z_T)