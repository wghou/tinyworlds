# Nano World Model

Inspired by Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT), and Google's Genie 1 Paper (https://arxiv.org/pdf/2402.15391).

The greatest challenge in training world models as opposed to video models is the requirement of action annotations at each timestep. WHen we require actions, we can no longer train on the entire internet's video data and get amazing results like Google's VEO3 (or, hint, Genie 3). 

Genie solves this problem by inferring the actions between frames, and labelling them in an unsupervised manner. This is the critical unlock to achieving scale with world models.

This codebase is meant to help people understand world modeling and a common architecture for doing so (likely semi-similar to Genie 3).

The core purpose of a world model is to predict the next state of some environment given the current state and some conditioning from an external entity(s). In predicting this next state, we encode the laws of how our environment changes from moment to moment.

We now want to learn a neural network which can map some image to some other image, conditioned on actions.

## Architecture 

This world model is autoregressive over discrete tokens, similar to LLMs. We can thus use many of the innovations from LLMs to improve our world model.

FSQ-VAE video tokenizer: Tokenize video into discrete tokens

FSQ Latent Action Model: Infer the discrete action between two frames

Dynamics Model: Given latent action and past frame tokens, predict latent next frame of video tokens

For all 3, we use the STTransformer, which we will go over first.


## ST Transformer
Papers: STTransformer(https://arxiv.org/pdf/2001.02908), FiLM (https://arxiv.org/pdf/1709.07871), RMSNorm(https://arxiv.org/pdf/1910.07467), SwiGLU(https://arxiv.org/pdf/2002.05202)

The ST Transformer consists of L spatial/temporal blocks, where each block contains a spatial attention layer, a temporal attention layer, and a feedforward layer.

In the spatial layer, attention operates over slices of 1 x P tokens, where P = Hp x Wp tokens within each timestep, where Hp = Pixel height / patch size (# patches along the H dimension) and Wp = Pixel width / patch size (# patches along the width dimension).

In the temporal layer, attention operates causally over slices of T x 1 across T timesteps for the same token. So a token in a given position attends to all other previous tokens in the exact same position but previous timesteps.

In the Feedforward Layer, we could have a basic Wx + b, ReLU, Wx + b, LayerNorm
However, it turns out that SwiGLU allows for greater model capacity and faster learning for the same number of parameters.
SwiGLU comes from Swish, which is x * sigmoid(x). SwiGLU adds a Gated Linear Unit (GLU), so w first compute x_t = Swish(W_1x + b) to W_2x + b, and then we have a final wx_t + b.

Both attentions and the feedforward have norms afterwards, either unconditioned (for Video Tokenizer and LAM Encoder) or conditioned (for LAM decoder and the dynamics model). 

For unconditioned STTransformer, we use RMSNorm, which computes norm as sqrt(eps + x / sum of x^2)

For conditioned STTransformer, we use Feature-wise Linear Modulation (FiLM). FiLM takes in the conditioning, in this case, actions for each timestep. It then uses a FeedForward Layer to transform each action latent into one beta vector and one gamma vector, both of embedding dim length. we then compute the norm as layernorm(x) * gamma + (1 + beta)

## Latent Action Model

The latent action model allows us to train without labels by learning unsupervised actions between to frames. We can then condition the dynamics on action without needing action labels for data

Encoder takes in all previous frames $(x_1...x_t)$ and next frame $x_{t + 1}$, outputs corresponding set of continuous latent actions $(a_1...a_t)$

Decoder takes in all previous frames and latent actions as input and predicts the next frame $x_{t + 1}$

To train model, use FSQ objective to limit number of predicted actions to small discrete set of codes

Limit vocab size $|A|$ of FSQ codebook (max number of possible latent actions)

Decoder only has access to history and latent action, $a_t$ should encode most meaningful change between past and future for decoder to successfully reconstruct future frame

Decoder gives training signal but is abandoned at inference time and replaced with user action

ST-transformer architecture for latent action model with causal mask in temporal layer to take entire video as input and generate actions between frames.

## Video Tokenizer

Compress videos into discrete tokens to reduce dimensionality and have higher quality video generation using FSQ VAE.

## Finite Scalar Quantization Variational Autoencoder (FSQ-VAE)
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


# Train Info

55M 16s video clips at 10FPS with 160 x 90 resolution
Final dataset 6.8M 16s video clips

11B parameter model

200M video tokenizer, patch size of 4, codebook with embedding size 32 and 1024 unique codes

LAM 300M, patch size 16, codebook embedding size 32, 8 unqiue codes (actions)


# Dynamics Model

At a high level, we want that at timestep $t \in [1, T]$, the dynamics model takes in tokenized video and latent action sequences from t=0 (defined as context-length frames old) up to $t - 1$ and predict next frame tokens $z_t$.

To get the action conditioning sequences, we run inference with the latent action model.

We condition on latent action sequences with Feature-wise Linear Modulation: conditioning is projected to gamma and beta, then we replace layernorm/RMSNorm in the transformer with layernorm(x) * gamma + (1 + beta)

We base the image prediction on MaskGIT. We randomly mask input tokens at train time according to bernoulli distribution with masking rate sampled uniformly from 0.5 to 1, and our goal is to predict the masked tokens.

At inference time, we append a fully masked frame, and have our model iteratively predict next frame tokens using MaskGIT Inference, which operates as follows:
For T steps:
1. Predict logits at each masked position and retrive token probabilities
2. Take the k most likely tokens of the unmasked positions and sample them, unmasking their positions
for k, we choose the cosine schedule such that the number of tokens sampled at each step increases exponentially.


# Inference Time
Paper: MaskGIT (https://arxiv.org/pdf/2202.04200)
Uses maskGIT inference

Player prompts model with initial frame, image tokenized with video encoder, then player specifies discrete latent action to take by choosing integer value in $[0, |A|]$, use to predict next frame

Repeat process autoregressively as actions passed to model and tokens decoded into video frames with decoder
