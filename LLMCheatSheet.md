
# LLM Cheat Sheet

## Pre-training loss functions

**Causal** language models: predict next word given previous words. Training loss is on all words in the block, attention masks are used so that Used in Attention is All You Need, GPT-x. 

**Masked** language models: Replace words with token [MASK], try to predict word using all words around that word. Used in BERT. 

**Permutation** language models: construct attention masks based on random permutations, predict words from randomly selected words around the word. Used in XLNet. 

**Replaced token detection**:  replace tokens with words generated from small generator network, predict whether tokens are replaced or not. Used in ELECTRA. 

## Architectures

### Attention layers

Attention layers perform the following operations. 
Unnormalized attention weight is computed as:
$$a_{ij}' = z_i^\top W_q^\top W_k x_j$$ 
Normalize with a soft-max:
$$Z_i = \sum_{j=1}^T \exp a_{ij}'$$
$$a_{ij} = SoftMax_j(a_{i\cdot}') = \frac{1}{Z_i} \exp a_{ij}'$$
Compute weighted sum of learned values:
$$h_{i} = \sum_{j=1}^T a_{ij} W_v x_v$$

## Self-attention layers

In self-attention, $z_i = x_i$. 

### Multi-head self-attention layers

We learn $A$ different query, key, and value matrices of size $H \times (H/A)$, and concatenate their outputs, where $H$ is the hidden size. 

### Encoder blocks

An encoder block consist of the following stacked layers:
* Multi-headed self-attention layer. $3A$ matrices of size $H \times (H/A+1)$. Number of parameters: $3 A H (H/A + 1) = 3(H^2 + AH)$.
* Concatenation then projection. One matrix of size $H \times H$. Number of parameters: $H^2$. 
* Residual connection. 
* Layer norm. Number of parameters: $2H$.
* Position-wise fully connected layer to hidden layer of size $FH$. One matrix of size $H \times (FH+1)$. Number of parameters: $H(FH+1)$. 
* Position-wise fully connected layer back to output of size $H$. One matrix of size $FH \times (H+1)$. Number of parameters: $FH(H+1)$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$. 
Total number of parameters per encoder block: 
$$3(H^2 + AH) + H^2 + 2H + H(FH+1) + FH(H+1) + 2H$$
$$= (3 + 1 + F + F)H^2 + (3A + 2 + 1 + F + 2)H$$
$$= (4+2F)H^2 + (3A + F + 5)H$$

### Decoder blocks
A decoder block consists of the following stacked layers:


See [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) for a good visualization, [The Annotated Tranformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) for sample code. 

**Encoder-decoder**: N encoding transformer blocks feeding into N decoding transfomer blocks. 

## Optimization

**RoBERTa**: BERT + optimization tweaks, bigger training data set
