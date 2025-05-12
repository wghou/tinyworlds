# Vector Quantized Variational Autoencoder (VQ-VAE)

Paper: https://arxiv.org/pdf/1711.00937

TLDR, in VQVAE:
1. Encoder-decoder with discrete representations/codes
2. Prior is learned rather than static

## Introduction

Maximum likelihood and reconstruction error are 2 most common unsupervissed model training methods for pixels

The best generative models (measured by log likelihood) have no latents and use a powerful decoder (ex PixelCNN)

They argue using discrete latents is potentially more natural, clearly for language, but they argue for:
1. speech (sequence of symbols) 
2. images (can be described by language)
3. reasoning/planning (also can be described by language)
Discrete latents are appealing since "powerful autoregressive models" (ex: GPTs) have already successful modeled distributions over discrete variables

## Benefits of VQ-VAE
1. Simple to train
2. Does not suffer from large variance
3. Avoids posterior collapse of most VAEs with powerful decoders
4. Similar performance to continuous VAEs
5. Conserves important features in latent space while optimising for maximum likelihood

## VAEs
Consist of:
1. Encoder network to parameterize posterior distribution $q(z | x)$ of discrete latent random variables z fiven input data x
2. Prior distribution $p(z)$
3. Decoder network to parameterize likelihood $p(x | z)$ over input data x given latent z

Notes:
1. Posteriors and priors assumed to be normal distribution and diagonal covariance matrix (each dimension is independent): enables gaussian reparametrization trick
2. Extensions: autoregressive prior/posterior models, normalising flows, inverse autoregressive posteriors

## VQVAE Architecture

Posterior and prior distributions are categorical, and samples drawn from these index into an embedding table, which are used as input to decoder network

Latent Embedding Space $e \in \mathbb{R}^{K \times D}$ where $K$ is the size of the discrete latent space and $D$ is the size of each latent embedding vector $e_i$, so there are $K$ embedding vectors of dimension $D$.

Input $x$ passed through encoder to produce $z_e(x)$, and discrete latent $z$ is calculated by nearest neighbor lookup with shared embedding space $e$, then corresponding embedding vector $e_k$ used as input to decoder network.

The posterior categorical distribution $q(z | x)$, which is the distribution of the latent $z$ which is the post-discretization encoder output, is defined one-hot as:

$q(z = k | x) = \begin{cases} 1 & \text{if } k = \text{argmin}_j ||e_k - z_e(x)||_2 \\ 0 & \text{otherwise} \end{cases}$

Essentially, We have a K-dimensional vector which has all 0s except a 1 for the vector in the latent embedding space which is closest to the vector produced by the encoder. 

-------- TODO: CLARIFY THIS SECTION --------

VQVAE is a VAE where we can bound $log p(x)$ with ELBO
Proposal distribution $q(z = k | x)$ is deterministic and with uniform prior over $z$, so $\sum_z q(z | x) \log \frac{q(z | x)}{p(z)} = \log \frac{1}{1 / K}$ since all other zs have 0 probability besides the deterministic z mapped to by the encoder, and p is uniform over K (so each $p(z)$ is 1/K), and the entropy of q is 0, thus we have KL divergence constant and equal to $0 - \log (1 / K) = \log K$

--------------------------------

## VQVAE Training

The discretization process has no real gradient, so VQVAE just copies the decoder input gradient ot the encoder output gradient

Loss Function has three terms:
1. Reconstruction loss $\log p(x | z_q(x))$ to optimze encoder/decoder (embeddings recieve no gradients from reconstruction loss so need another loss for embedding space)
2. Vector Quantization loss $\|\text{sg}[z_e(x)]- e\|_2^2$ with $l_2$ error to move embedding vectors $e_i$ towards encoder outputs $z_e(x)$
3. Commitment Loss $\beta \|z_e(x) - \text{sg}[e]\|_2^2$ to ensure encoder commits to embedding and output does not grow (since size of embedding space is unbounded), thus pulls encoder outputs toward discrete embeddings

Total training objective:
$L = \log p(x | z_q(x)) + \|\text{sg}[z_e(x)]- e\|_2^2 + \beta \|z_e(x) - \text{sg}[e]\|_2^2$

Where sg is the stopgradient operator that is identity at forward computation time and has zero partial derivatives, and thus constrains the operand to be non-updated constant

Encoder optimized first and last terms, embeddings optimize middle term, and decoder optimizes first term only.

In practice, for images we use N discrete latents (ex: field of 32 x 32 latents for ImageNet).

Log likelihood of complete model $p(x)$ is:

$\log p(x) = \log \sum_k p(x | z_k) p(z_k)$

## VQVAE for Video

Can model 128 x 128 x 3 images by compressing to 32 x 32 x 1 discrete space with K = 512

Initial 6 frames input to model and 10 frames sampled from VQ-VAE with all actions set to forward and right

Generation is purely in latent space $z_t$ without need to generate actual imgaes themselves

Each image in sequence $x_t$ created by mapping latents with deterministic decoder to pixel space after all latens generated using prior model $p(z_1,...,z_T)

VQ-VAE used to imagine long sequences purely in latent space, also works for model without actions