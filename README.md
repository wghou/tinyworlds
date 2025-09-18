![image](assets/tinyworlds.png)

TinyWorlds is a minimal autoregressive World Model built from Google's [Genie 1 Paper](https://arxiv.org/pdf/2402.15391).

World models, over video models, need action data in addition to raw video data. 

This constraint means we can no longer train on the entire internet's video data to get consistent world states like in Google's [VEO3](https://deepmind.google/models/veo/). 

Genie solves this problem by inferring the actions between frames, with **no labels**.

This unsupervised action generation is likely the critical unlock to achieving scale with world models like in [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/).

This minimal codebase, TinyWorlds, is meant to help people understand world modeling and the clever autoregressive, unsupervised method Genie used to achieve **scalable world models**.


## Installation

```bash
git clone https://github.com/AlmondGod/nano-genie.git

cd nano-genie

pip install -r requirements.txt

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

export PYTHONPATH="/workspace/nano-genie:$PYTHONPATH"

python scripts/full_train.py --config configs/training_config.yaml
```

## Architecture 

TinyWorlds uses an autoregressive world model over discrete tokens, so we can use SOTA LLM techniques to improve our world model. 

Why discrete tokens? Discretization makes our dynamics prediction problem much easier, because instead of predicting an image a near-infinite continuous space, it need only select one of the ~4096 tokens in our vocabulary.

Our world model consists of three modules:

# TODO: insert architecture diagram

**Video Tokenizer:** Continuous domains like audio and video require more clever tokenization than inherently discrete language. We tokenize using a VAE: we train a model to reconstruct a sequence of video, placing a small discrete bottleneck (our video tokens) in model's center which should capture the important information in the video.

**Action Tokenizer:** This infers the discrete action token between two frames. We again use a VAE to reconstruct the next frame conditioned on the previous frame and a discrete bottleneck (our action token) that encodes information of the transition between previous and next frame.

**Dynamics Model:** Given action and past frame tokens, this predicts our next frame tokens. This is the core of our world model that captures the structure and emergent phenomena of our tiny video game worlds.

For all 3, I used STTransformer, and for the Tokenizer and LAM, I used FSQVAE.

## Space-Time Transformer (STT)
papers: [STTransformer](https://arxiv.org/pdf/2001.02908), [FiLM](https://arxiv.org/pdf/1709.07871), [RMSNorm](https://arxiv.org/pdf/1910.07467), [SwiGLU](https://arxiv.org/pdf/2002.05202)

The Space-Time Transformer consists of B spatial/temporal blocks, where each block contains a spatial attention layer, a temporal attention layer, and a feedforward layer. For a brush up on regular self-attention, see Karpathy's [GPT From Scratch Video](https://youtu.be/kCc8FmEb1nY?si=tvfcBnGHBbEiS70v&t=3748).

In the spatial layer, each token attends to all other tokens in the same frame (within its timestep). Attention operates over a given timestep with P tokens, where P = Hp x Wp, Hp = Pixel height / patch size (# patches along the H dimension), and Wp = Pixel width / patch size (# patches along the width dimension). 

In the temporal layer, each token in a given position attends causally to other previous tokens in the exact same position but previous timesteps. Attention operates causally over slices of T x 1 across T timesteps for the same token.

In the Feedforward Layer, we could use a basic FFL: Wx + b -> ReLU -> Wx + b -> LayerNorm. However, it turns out that SwiGLU allows for greater model capacity and faster learning for the same number of parameters.
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
Paper: [Finite Scalar Quantization](https://arxiv.org/pdf/2309.15505), [VQ-VAE](https://arxiv.org/pdf/1711.00937)

TLDR, in FSQ:
1. Encoder-decoder use discrete representations/codes
3. We recieve direct gradients for the discretization and require no auxiliary losses, unlike in VQVAE

## Introduction

VQVAE paper argues using discrete latents is potentially more natural, clearly for language, but they argue even for images, which can be described by language.

Discrete latents are appealing since "powerful autoregressive models" (ex: GPTs) have already successful modeled distributions over discrete variables

## VAEs
Paper: [Overview of VAEs](https://arxiv.org/pdf/1906.02691), Eric Jang's [Variational Methods](https://blog.evjang.com/2016/08/variational-bayes.html)

VAEs consist of:
1. Encoder network to parameterize posterior distribution $q(z | x)$ of latent random variables z given input data x (probability image is a cat given a frame of a cat)
2. Unknown prior distribution $p(z)$ (probability any random thing is a cat)
3. Decoder network to parameterize likelihood $p(x | z)$ over input data x given latent z (frame of cat given idea of cat)

We cannot learn these quantities directly, but we can instead learn to maximize the likelihoods of z | x and x | z by ascending the reconstruction objective p(x | z | x), where z will learn semantically meaningful information because we constrain its dimensionality be low (it is forced to choose only the most important information from an image).

## Finite Scalar Quantization

Posterior and prior distributions are categorical, and samples drawn from these fall into a specific region of a hypercube, which are used as discrete inputs to decoder network.

Latent Embedding Space $|e| = {L^{D}}$ where $L$ is the number of levels/bins per dimension and $D$ is the dimensionality of the hypercube (number of dimensions in the latent embeddings). Essentially, for each dimension, we can fall into one of L buckets, and we have D dimensions. if we had 3 dimensions and 2 levels per dimension, we'd have 2^8 possible regions in the cube.

Input $x$ passed through encoder to produce $z_e(x)$, and discrete latent $z$ is calculated by:

1. tanhing to [-1,1]
2. transforming to [0, L]
3. rounding to the nearest integer
4. transforming back to [-1,1]

the resulting discrete latent vector $e_k$ is used as input to decoder network.

## FSQVAE Training

In FSQ, the discretization process has a straight-through gradient, where the input to the decoder is equal to z + stopgrad(z_q - z). So the quantity is z_q, but the gradient is on z.

We thus need no extra loss terms unlike in VQVAE, which makes this method siginificantly more wieldy.

stopgrad is the stopgradient operator (in pytorch, .detach()) that is identity at forward computation time and has zero partial derivatives, and thus constrains the operand to be non-updated constant.


## Action Tokenizer

The Action Tokenizer allows us to train without labels by learning unsupervised actions between to frames. We can then condition the dynamics on action without needing action labels for data

The LAM Encoder takes in a sequence of frames $(x_1...x_t+1)$, and outputs a corresponding set of continuous action latent vectors between the frames $(a_1...a_t)$, where a_1 is the action taken between x_1 and x_2.

To create the discrete action codebook, we again use an FSQ objective to bound and bin the continuous action latent vectors outtputed by the encoder into one of |codebook size| cubes which represent our codebook.

The LAM Decoder takes in all previous frames $(x_1...x_t)$ and quantized action latent vectors $(x_1...x_t)$ as input and predicts the next frame $x_{t + 1}$.

Since the decoder only has access to frame history and the action token, $a_t$ should encode most meaningful change between the past frame and the future frame for decoder to successfully reconstruct future frame. 

In practice, the decoder tends to try to ignore actions as much as possible. To counteract this, we mask most frames except the first, so the decoder must learn to use the string of actions as signal. We also add auxiliary variance regularization batch-wise to the encoder.

At inference time, we only use the learned cubes (latents) which correspond to action indices that the user can output. These actions should each end up corresponding to a semantically meaningful condition for next frame prediction.

# Dynamics Model
Additional paper: MaskGIT (https://arxiv.org/pdf/2202.04200)

At a high level, we want that at timestep $t \in [1, T]$, the dynamics model takes in tokenized video and action sequences from t=0 (defined as context-length frames old) up to $t - 1$ and predict next frame tokens $z_t$.

In practice, we use a method similar to MaskGIT and BERT: we mask a subset of tokens, and train our model to predict the masked tokens conditioned on all current and previous frame tokens. The prediction is conditioned on action tokens inferred at train time by the action tokenizer.

For inference, at each step, we append a fully masked frame to our context sequence, then our model iteratively predicts next frame tokens using MaskGIT Inference as follows:
For T steps:
1. Predict logits at each masked position and compute token probabilities
2. Take the k most likely tokens out of the still unmasked positions and sample them, unmasking their positions
We choose k using the exponential schedule (first step would sample ~1 token, then ~2, then ~5, then ~20, then ~50, etc)


# Full TinyWorlds Inference

We first give the model an initial frame from the training distribution and tokenize the image with our video tokenizer
We then run the following loop:
3. The player specifies one of the n_actions action tokens to use by choosing integer value in $[0, |A|]$
4. Condition the dynamics model with context window c on the video tokens t-c...t and action tokens t-c..t and run dynamics inference 
5. Detokenize the predicted video tokens into a new video frame for the user

We repeat process autoregressively over the time dimension as actions are passed to model and tokens are predicted by the dynamics model and detokenized into frames to display to the user.

# Data

The data is processed and downsampled from mp4s into hdf5 files. You can download the following datasets I uploaded to huggingface ([Datasets](https://huggingface.co/datasets/AlmondGod/tinyworlds), [Pretrained models](https://huggingface.co/AlmondGod/tineyworlds-models))

1. PicoDoom (`picodoom_frames.h5`)
2. Pong (`pong_frames.h5`)
3. Zelda Ocarina of Tima (`zelda_frames.h5`)
4. Pole Position (`pole_position_frames.h5`)
5. Sonic (`sonic_frames.h5`)

Any data can be added by creating a new dataclass and specifying the mp4 path in [datasets.py](datasets/datasets.py)

```bash
# retrieve data and downlaod into data/
python scripts/download_assets.py datasets --pattern "sonic_frames.h5"

# puls sonic dynamics checkpoint into results/<TIMESTAMP>_sonic_models/dynamics/checkpoints
python scripts/download_assets.py models --type dynamics --suite-name sonic_models
```

# Development Process and Decisions

I originally used VQVAE for both the video tokenizer and the action tokenizer as genie 1 originally used. VQVAE is sometimes unwieldy, for it has 2 auxiliary losses which require careful tuning, and often doesn't result in codebook usage higher than 20% if even.

FSQ came forward as a cleaner alternative to VQVAE.

Conditioning in genie 1 has the action embeddings added to the video embeddings. I found that using FiLM from the action latents both allowed for independent variance of video and action latent space dimensions, and allowed for stronger care for action latents.

The greatest challenge was avoiding action tokenizer collapse, which was solved by
1. Switching VQVAE (tended to collapse to 1/8 codes quickly) to FSQVAE
2. Using only the first frame and masking all others
3. Adding low-weight encoder variance loss (across the batch dim)

I found RMSNorm better than layernorm in ablations (TODO: do full ablation run and loss comparison)

I found SwiGLU better than ReLU and SiLU (TODO: full ablation run/loss comparison)

The default model uses 4 transformer blocks, with d_model 128, 8 heads, 256 FFN hidden dim, and 4-frame sequences which make for around 1.3M parameter tokenizers and dynamics model.

# Shape Annotation Key

B: batch size \
T: time/sequence dimension (number of frames) \
P: number of patches \
E: embedding dim \
L: Video Tokenizer latent dim \
A: LAM latent dim (action dim) \
D: number of bins for each video tokenizer dim \
L^D: Size of the video tokenizer codebook \
C: image channels \
H: pixel height \
W: pixel width \
Hp: patch height \
Wp: patch width \
S: patch size

# Training and Inference Options

I added support for:
1. torch compile which allows us to use faster kernels for certain pre-optimized operations
2. distributed data parallel (DDP) which allows us to train using multiple gpus by using different data per-gpu
3. automatic mixed precision (AMP) which dynamically switches between FP32 and BF16 based on the current nodes used floating point range
4. FP32 training which lets us use nvidia floating point 32 for extremely precise floating point operations 
(all of the above were made much easier by torch, thank you torch team)

# An Appreciation of World Models
A world model predicts the next state of an environment given current state and some conditioning. 

To predict the next world state, we encode the structure and emergent phenomena of the universe itself.

We train a deep network which, given an image and action, predicts the most likely next image.

World models can both act as cortexes to give physical world understanding to models and as simulators for models and humans to experience new structures of reality.


# Next Steps

- [ ] Implement MoE in the Feedforward layer
- [ ] Try RoPE/AliBi Spatial/Temporal Position Embeddings
- [ ] Scale! Train on more GPUs and scale to multibillions of params by adding FSDP Support
- [ ] Add more datasets (Terraria, Street Fighter, your favorite retro videogame!) 
- [ ] Replace the mean pool + concat in the action tokenizer with attention pooling (t to t + 1 then mean pool)
- [ ] Accelerate dynamics training by producing, saving, and loading pre-processed tokens instead of full frames 
- [ ] Try different optimizers (Muon, SOAP)

**Please make a PR! There are many small things to try which could offer massive performance gains, and the codebase is meant to be built upon**