![image](assets/tinyworldsbannerv3.png)

TinyWorlds is a minimal autoregressive world model built on Google Deepmind's [Genie Architecture](https://arxiv.org/pdf/2402.15391).

Requiring both video and action data, world models can't use internet video (which has no actions) to scale like Google's [VEO3](https://deepmind.google/models/veo/).

Google's [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/) solves this by inferring the actions between frames using **no prior action data**.

TinyWorlds is meant to help people understand world modeling and the clever autoregressive, unsupervised method Deepmind used to achieve **scalable world models**.

## Table of Contents

- [Getting Started](#getting-started)
- [Why World Models?](#why-world-models)
- [Overview](#architecture-overview)
- [Building Blocks](#architecture-building-blocks)
   - [Space-Time Transformer (STT)](#space-time-transformer-stt)
   - [Variational Autoencoder (VAE)](#vaes)
   - [Finite Scalar Quantization (FSQ)](#finite-scalar-quantization)
- [Architecture](#architecture)
   - [Video Tokenizer](#video-tokenizer)
   - [Action Tokenizer](#action-tokenizer)
   - [Dynamics Model](#dynamics-model)
   - [TinyWorlds Inference](#full-tinyworlds-inference)
   - [Data](#data)
   - [Training and Inference Acceleration](#training-and-inference-acceleration-options)
   - [Shape Annotation Key](#shape-annotation-key)
- [Next Steps](#next-steps)

# Getting Started

```bash
# Installation
git clone https://github.com/AlmondGod/nano-genie.git
cd nano-genie
pip install -r requirements.txt
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export PYTHONPATH="/workspace/nano-genie:$PYTHONPATH"

# Training
# 1. download data (pong)
python scripts/download_assets.py datasets --pattern "pong_frames.h5"
# 2. run training
python scripts/full_train.py --config configs/training_config.yaml

# Inference
# 1. pull pretrained sonic checkpoints from huggingface
python scripts/download_assets.py models --suite-name sonic
# 2. run inference
python scripts/run_inference.py --config configs/inference.yaml --use-latest-checkpoints
```

# Why World Models?
A world model is a function mapping the current state of the environment, plus conditioning, to next state. 

In current world models, we train a deep network which, given an image and action, predicts the most likely next image. In the near term, world models can be:
1. cortexes to give physical world understanding to models 
2. simulators for models to interact with physics fully-online 
3. experiences with new structures of reality for humans to interact with

To predict the next world state accurately, the world function must compress all the information in the world into a set of laws. 

Hence the name world model: **it is capturing all the inherent structure and emergent phenomena of the world.** 

A world model is the world itself. The universe is a world model. Reality emerges within that model.

We are only at the beginning of world modeling. Are you getting it?

# Overview 

![tinyworldsarch](/assets/tinyworldsarchv3.png)

TinyWorlds uses an autoregressive transformer world model over discrete tokens, so we can use SOTA LLM techniques to improve our world model. 

Why discrete tokens? Discretization makes our dynamics prediction problem much easier, because instead of predicting an image a near-infinite continuous space, it need only select one of the ~1000 tokens in our vocabulary (aka codebook).

TinyWorlds consists of three modules:

**Video Tokenizer:** This tokenizer (a VAE) reconstructs a sequence of video with a small discrete bottleneck (our video tokens) in the middle. This layer should maximally compress the important information in the video.

**Action Tokenizer:** This tokenizer (also a VAE) infers the discrete action token between two frames. It trains by reconstructing the next frame using the previous frame and a discrete action token that sees the next frame.

**Dynamics Model:** Given past action and frame tokens, this predicts our next frame tokens. This should capture the physics and emergent phenomena of our tiny video game worlds.

# Building Blocks

The above modules are built using [ST Transformer](https://arxiv.org/pdf/2001.02908), Finite Scalar Quantization, and VAE, explained below:

### Space-Time Transformer (STT)
![stt](/assets/spacetimetransformer.png)

Each Space-Time Transformer block contains a spatial attention layer, a temporal attention layer, and a FeedForward Network (FFN). For a brush up on self-attention, see Karpathy's [GPT From Scratch Video](https://youtu.be/kCc8FmEb1nY?si=tvfcBnGHBbEiS70v&t=3748)

In the spatial layer, each token attends to all other tokens in the same frame (within its timestep). 
In the temporal layer, each token in a given position attends causally to other previous tokens in the exact same position but previous timesteps.

The FFN is a 2-layer MLP on each embedding vector. Inspired by divine benevolence, I used [SwiGLU](https://arxiv.org/pdf/2002.05202) for the FFN. SwiGLU adds a Gated Linear Unit (GLU) to [Swish](https://arxiv.org/pdf/1710.05941), and is computed as $x_t = W_3[Sigmoid(W_1x + b1) * W_2x + b2] + b3$.

Each attention/FFN have norms afterward, either unconditioned (Video Tokenizer, Action Encoder) or conditioned on actions (Action Decoder, Dynamics). 

For unconditioned STTransformer, we use [Root Mean Squared Normalization (RMSNorm)](https://arxiv.org/pdf/1910.07467), where we divide our input by $\sqrt(\epsilon + x / \sum x^2)$. RMS is less sensitive to extreme outliers than simply dividing by variance.

For conditioned STTransformer, we use [Feature-wise Linear Modulation (FiLM)](https://arxiv.org/pdf/1709.07871). FiLM passes actions for each timestep through an FFN to transform each action latent into Gamma ($\gamma$) and Beta ($\beta$). Our norm is then $(x - \mu) / \sigma * (1 + \gamma) + \beta$

### Variational Autoencoder (VAE)
[Overview of VAEs](https://arxiv.org/pdf/1906.02691), Eric Jang's [Variational Methods](https://blog.evjang.com/2016/08/variational-bayes.html)

*\*VAEs are complex but an overview with many details omitted is below*

VAEs are defined by:
1. An encoder network to parameterize the approximate posterior distribution $q(z | x)$ of latent variables $z$ given data $x$
2. Prior distribution $p(z)$, chosen as regular gaussian $N(0,I)$
3. A decoder network to parameterize likelihood $p(x | z)$ over input data x given latent z

We maximize $log(p(x | z))$, the likelihood we exactly reconstruct the input x given our latent z. 

The important takeaway is that $z$ is low dimensional, so for reconstruction, it will compress all the important information from $x$.

### Finite Scalar Quantization (FSQ)

![fsq](/assets/finitescalarquantizer.png)

Since we want a set of discrete tokens, we need to quantize z (use one of a finite set of possible zs).

Thinking of vectors as points in high dimensional space, FSQ is a quantization method that divides space into hypercubes. The hypercube a vector falls into becomes its quantized representation.

Concretely, we quantize a vector in FSQ by:

1. tanh(x) which bounds to [-1,1]
2. scale/shift to [0, L]
3. rounding to the nearest integer (quantization step)
4. scale/shift back to [-1,1]

The token vocabulary has size ${L^{D}}$ where $L$ is bins per dimension and $D$ is the dimensionality of the hypercube. With 3 dimensions and 2 levels per dimension, we'd have 8 possible regions in the cube and a size 8 token vocabulary. 

FSQ VAEs let us learn structured hypercubes to use as our token vocabularies that encode information about the input. In our context, maybe one of these hypercubes represents moving left, another jumping, another crouching, et cetera.

To allow gradients to flow to the encoder (since quantization is non-differentiable), we pass the post-quantization gradients directly to the pre-quantization layer. 

Precisely, the decoder takes as input $z + stopgrad(z_q - z)$ where stopgrad is, in pytorch, `.detach()`. The decoder only uses $z_q$ (since $z - z = 0$), but the gradient is taken only on $z$.

Now we can go over each module:

# Architecture

## Video Tokenizer

![videotokenizer](/assets/videotokenizer.png)

The video tokenizer compresses videos into discrete tokens to reduce the dimensionality of the dynamics model but keep high quality video generation.

It uses an FSQ VAE implemented with an STTransformer (causal over time). Each video token contains information about both its own patch and other patches in the same location or timestep.

## Action Tokenizer

![actiontokenizer](/assets/actiontokenizer.png)

The Action Tokenizer is the key to scalability: it allows us to train without action labels by learning to infer actions between to frames. We then condition the dynamics on these actions.

It is also an FSQ VAE. The encoder takes in a sequence of frames $(x_1...x_t-1)$, and outputs action tokens between the frames $(a_1...a_t-1)$, where a_1 is the action taken between x_1 and x_2.

The Action Tokenizer Decoder takes in all previous frames $(x_1...x_t-1)$ and quantized action latent vectors $(a_1...a_t-1)$ as input and predicts the next frames $(x_2...x_t)$.

The action tokens should learn to encode the most meaningful change between the past and current frame, which should correspond to some high-level action.

In practice, the action decoder tries to ignore actions and infer purely from images. To counteract this, 
1. we mask most frames except the first, so the decoder must learn to use the string of actions as signal for reconstruction
2. we encourage batch-wise variance in the encoder through an auxiliary loss

At inference time, we map each key to one of the action tokens that conditions the dynamics for the user to influence video generation.

## Dynamics Model
![dynamicsmodel](/assets/dynamicsmodel.png)

At timestep $t$, the dynamics model should take in tokenized video tokens $z_{1..t - 1}$ and action tokens $a_{_1..t - 1}$ and predict next frame tokens $z_{t}$.

In practice, we train dynamics like [MaskGIT](https://arxiv.org/pdf/2202.04200) and [BERT](https://arxiv.org/pdf/1810.04805).

We mask a subset of tokens and train our model to predict the masked tokens, conditioned on all current and previous frame and action tokens.

To infer dynamics at a given step, we first append a fully masked frame to our context sequence. Then, for T steps we:
1. Predict logits at each masked position
2. Compute token probabilities with softmax
3. Sample the k most likely tokens out of the still unmasked positions
4. Place them into the context tensor, removing corresponding mask tokens
5. Repeat

To choose k, we use an exponential schedule (first step samples ~1 token, then ~2, then ~5, then ~20, then ~50, etc)

## TinyWorlds Inference

Given initial context frames from the training distribution, we first tokenize them.

We then run the following loop:
1. The player specifies one of the n_actions action tokens to use by choosing integer in $[0, |A|]$
2. Condition the dynamics model with context window c on the video tokens t-c...t and action tokens t-c..t and run dynamics inference 
3. Detokenize the predicted video tokens into a new video frame for the user

We repeat this process autoregressively over the time dimension as actions are passed to the model, tokens are predicted by the dynamics model, we detokenize them into frames to display to the user.

This process also lets us predict multiple future frames at once (bounded by memory and the training distribution), which can improve inference quality.

## Data

![datasets](/assets/datasets_stylized.png)

The data is processed and downsampled from gameplay mp4s into hdf5 files. You can download the above datasets from [Huggingface TinyWorlds Datasets](https://huggingface.co/datasets/AlmondGod/tinyworlds/tree/main) with 

```bash
python scripts/download_assets.py datasets --pattern "<name>_frames.h5`
```

Available are:
1. **PicoDoom** (`picodoom_frames.h5`), a minimal version of Doom
2. **Pong** (`pong_frames.h5`), the classic
3. **Zelda Ocarina of Time** (`zelda_frames.h5`), one of the originl 2D Zelda games
4. **Pole Position** (`pole_position_frames.h5`), a pixel racing game
5. **Sonic** (`sonic_frames.h5`), the original game

Any new data can be added by creating a new dataclass in [datasets.py](datasets/datasets.py).

## Training and Inference Acceleration

TinyWorlds supports the following torch features to accelerate training and/or inference:
1. **Torch compile**, which allows us to use faster CUDA kernels for certain pre-optimized operations like attention and matmuls
2. **Distributed data parallel (DDP)**, which allows us to train using multiple gpus by using same model different data per-gpu
3. **Automatic mixed precision (AMP)**, which scales certain ops from FP32 to BF16 based on the current nodes used floating point range
4. **TF32 training**, which lets us use NVIDIA TensorFloat32 for tensor-core-optimized matmuls and convolutions

## Shape Annotation Key

All tensors are shape-annotated and use einops tensor manipulation operations with the following abbreviations:

**B:** batch size \
**T:** time/sequence dimension (number of frames) \
**P:** number of patch tokens per frame \
**E:** embedding dim (d_model) \
**L:** Video Tokenizer latent dim \
**A:** Action Tokenizer latent dim (action dim) \
**D:** number of bins for each video tokenizer dim \
**L^D:** Size of the video tokenizer vocabulary \
**C:** image channels \
**H:** pixel-grid height \
**W:** pixel-grid width \
**Hp:** patch-grid height \
**Wp:** patch-grid width \
**S:** patch size

# Next Steps

- [ ] Implement MoE in the Feedforward layer
- [ ] Try RoPE/AliBi Spatial/Temporal Position Embeddings
- [ ] Scale! Train on more GPUs and scale to multibillions of params by adding FSDP Support
- [ ] Add more datasets (Terraria, Street Fighter, your favorite retro videogame!) 
- [ ] Replace the mean pool + concat in the action tokenizer with attention pooling (t to t + 1 then mean pool)
- [ ] Accelerate dynamics training by producing, saving, and loading pre-processed image patch embeddings instead of full frames 
- [ ] Try different optimizers (Muon, SOAP)
- [ ] Try [AdaLN-Zero](https://arxiv.org/pdf/2212.09748) instead of FiLM (adds a pre-scale parameter)
- [ ] Add new schedulers for MaskGIT like cosine and [Halton](https://github.com/valeoai/Halton-MaskGIT)

**Please make a PR! I've added TODOs throughout and there are many small things to try which could offer massive performance gains. The codebase is meant to be built upon.**

*aesthetic inspired by [Tinygrad](https://tinygrad.org/) and [Tinygpu](https://github.com/adam-maj/tiny-gpu)*