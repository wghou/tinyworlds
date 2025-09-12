# Nano World Model

Inspired by Karpathy's NanoGPT (https://github.com/karpathy/nanoGPT), and Google's Genie 1 Paper (https://arxiv.org/pdf/2402.15391).

The greatest challenge in training world models as opposed to video models is the requirement of action annotations at each timestep. WHen we require actions, we can no longer train on the entire internet's video data and get amazing results like Google's VEO3 (or, hint, Genie 3). 

Genie solves this problem by inferring the actions between frames, and labelling them in an unsupervised manner. This is the critical unlock to achieving scale with world models.

This codebase is meant to help people understand world modeling and a common architecture for doing so (likely semi-similar to Genie 3).

The core purpose of a world model is to predict the next state of some environment given the current state and some conditioning from an external entity(s). In predicting this next state, we encode the laws of how our environment changes from moment to moment.

We now want to learn a neural network which can map some image to some other image, conditioned on actions.

This can be used for training robotics models, for explicitly giving models understanding of the physical world, and for simulating new worlds (eventually full universes) we want to experience.

## Installation

```bash
git clone https://github.com/AlmondGod/nano-genie.git

cd nano-genie

pip install -r requirements.txt

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python train_full_pipeline.py --config configs/pipeline.yaml
```

## Architecture 

This world model is autoregressive over discrete tokens, similar to LLMs. We can thus use many of the innovations from LLMs to improve our world model. Discretization makes our dynamics prediction problem much easier, because instead of prediction in an infinite continuous space, the dynamics model knows its outputting one of the 1024 tokens in our vocabulary.

FSQ-VAE video tokenizer: This creates our vocabulary. In LLMs we can use the Byte-pair Encoding algorithm which merges symbols to maximize compression since language is inherently discrete, but continuous domains like audio and video require more clever tokenization. We tokenize by training a model to reconstruct a sequence of video, and place a small discrete bottleneck in the middle of the model which should learn to capture the most information contained in the base video.

FSQ Latent Action Model: This infers the discrete action between two frames. Similarly to our video tokenizer, we do so by reconstructing the next frame conditioned on a discrete bottleneck of actions, and our uncompressed previous frame.

Dynamics Model: Given latent action and past frame tokens, predict latent next frame of video tokens. This is the core of our world model that captures the dynamics of the video we give.

For all 3, we use the STTransformer, which we will go over first.

## Space-Time Transformer (STT)
Papers: STTransformer(https://arxiv.org/pdf/2001.02908), FiLM (https://arxiv.org/pdf/1709.07871), RMSNorm(https://arxiv.org/pdf/1910.07467), SwiGLU(https://arxiv.org/pdf/2002.05202)

The Space-Time Transformer consists of B spatial/temporal blocks, where each block contains a spatial attention layer, a temporal attention layer, and a feedforward layer. For a brush up on regular self-attention, look here(TODO: add best link).

In the spatial layer, each token attends to all other tokens within its timestep. Attention operates over a given timestep with P tokens, where P = Hp x Wp, Hp = Pixel height / patch size (# patches along the H dimension), and Wp = Pixel width / patch size (# patches along the width dimension). 

In the temporal layer, each token in a given position attends causally to other previous tokens in the exact same position but previous timesteps. Attention operates causally over slices of T x 1 across T timesteps for the same token.

In the feedforward Layer, we could use a basic FFL: Wx + b -> ReLU -> Wx + b -> LayerNorm. However, it turns out that SwiGLU allows for greater model capacity and faster learning for the same number of parameters.
SwiGLU comes from Swish, which is x * sigmoid(x). SwiGLU adds a Gated Linear Unit (GLU), so we first compute x_t = Swish(W_1x + b) to W_2x + b, and then we have a final wx_t + b.

Both attentions and the feedforward have norms afterward (postnorm), either unconditioned (for Video Tokenizer and LAM Encoder) or conditioned on actions (for LAM decoder and the dynamics model). 

For unconditioned STTransformer, we use RMSNorm, which computes norm as sqrt(eps + x / sum of x^2)

For conditioned STTransformer, we use Feature-wise Linear Modulation (FiLM). FiLM takes in the conditioning, in this case, actions for each timestep. It then uses a FeedForward Layer to transform each action latent into one beta vector and one gamma vector, both of embedding dim length. we then compute the norm as layernorm(x) * gamma + (1 + beta)


## Video Tokenizer

The video tokenizer compresses videos into discrete tokens to reduce dimensionality and have higher quality video generation.
It does so as an FSQVAE implemented with an STTransformer that attends to tokens full-spatially and temporal-causally. 
Thus, each token contains information about its frame in relation to itself and to previous frames. 
Here is how FSQVAE works:

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


## Latent Action Model (LAM)

The Latent Action Model allows us to train without labels by learning unsupervised actions between to frames. We can then condition the dynamics on action without needing action labels for data

The LAM Encoder takes in a sequence of frames $(x_1...x_t+1)$, and outputs a corresponding set of continuous latent actions between the frames $(a_1...a_t)$, where a_1 is the action taken between x_1 and x_2.

The LAM Decoder takes in all previous frames $(x_1...x_t)$ and latent actions $(x_1...x_t)$ as input and predicts the next frame $x_{t + 1}$.

To create the discrete bottleneck that becomes our action vocabulary, we again use an FSQ objective to limit number of predicted actions to small discrete set of codes that is the number of actions we want to be able to take. 

Since the decoder only has access to frame history and the latent action, $a_t$ should encode most meaningful change between the past frame and the future frame for decoder to successfully reconstruct future frame. 

Decoder gives training signal but is abandoned at inference time and replaced with user action

# Dynamics Model

At a high level, we want that at timestep $t \in [1, T]$, the dynamics model takes in tokenized video and latent action sequences from t=0 (defined as context-length frames old) up to $t - 1$ and predict next frame tokens $z_t$.

To get the action conditioning sequences, we run inference with the latent action model.

We condition on latent action sequences with Feature-wise Linear Modulation: conditioning is projected to gamma and beta, then we replace layernorm/RMSNorm in the transformer with layernorm(x) * gamma + (1 + beta)

We base the image prediction on MaskGIT. We randomly mask input tokens at train time according to bernoulli distribution with masking rate sampled uniformly from 0.5 to 1, and our goal is to predict the masked tokens.

# Inference Time
Paper: MaskGIT (https://arxiv.org/pdf/2202.04200)

At inference time, we append a fully masked frame, and have our model iteratively predict next frame tokens using MaskGIT Inference, which operates as follows:
For T steps:
1. Predict logits at each masked position and retrive token probabilities
2. Take the k most likely tokens of the unmasked positions and sample them, unmasking their positions
for k, we choose the cosine schedule such that the number of tokens sampled at each step increases exponentially.

Uses maskGIT inference

Player prompts model with initial frame, we tokenize the image with our video tokenizer, then the player specifies one of the n_actions discrete latent actions to take by choosing integer value in $[0, |A|]$, we use that index to access the corresponding latent, and then condition the dynamics model with context window c on the video tokens t-c...t and latent actions t-c..t using the maskgit inference process. 

We repeat process autoregressively over the time dimension as actions are passed to model and tokens are predicted by the dynamics model and detokenized into frames to display to the user.

# Data

THe data is downsampled from youtube videos. Currently we have:
1. PicoDoom
2. Pong
3. Zelda Ocarina of Tima
4. Pole Position
5. Sonic
