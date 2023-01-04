
# LLM Cheat Sheet

## Self-supervised loss functions

**Causal** language models: predict next word given previous words. Training loss is on all words in the block, attention masks are used so that Used in Attention is All You Need, GPT-x. 

**Masked** language models: Replace words with token [MASK], try to predict word using all words around that word. Used in BERT. 

**Permutation** language models: construct attention masks based on random permutations, predict words from randomly selected words around the word. Used in XLNet. 

**Replaced token detection**:  replace tokens with words generated from small generator network, predict whether tokens are replaced or not. Used in ELECTRA. 

## Architecture components

### Attention layers

Attention layers perform the following operations. 
Unnormalized attention weight is computed as:
$$a_{ij}' = z_i^\top W_q^\top W_k x_j$$ 
Normalize with a soft-max:
$$Z_i = \sum_{j=1}^T \exp a_{ij}'$$
$$a_{ij} = SoftMax_j(a_{i\cdot}') = \frac{1}{Z_i} \exp a_{ij}'$$
Compute weighted sum of learned values:
$$h_{i} = \sum_{j=1}^T a_{ij} W_v x_v$$

### Self-attention layers

In self-attention, $z_i = x_i$. 

### Multi-head self-attention layers

We learn $A$ different query, key, and value matrices of size $H \times (H/A)$, and concatenate their outputs, where $H$ is the hidden size. 

### Encoder blocks

An encoder block inputs a sequence (derived from the word tokens and their position embedding) and consists of the following stacked layers:
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
A decoder block inputs a target sequence and a memory sequence, possibly from an encoder block. It consists of the following stacked layers:
* Multi-head self-attention layer on the target sequence, using an attention mask to prevent using future information. $3A$ matrices of size $H \times (H/A+1)$. Number of parameters: $3 A H (H/A + 1) = 3(H^2 + AH)$.
* Concatenation then projection. One matrix of size $H \times H$. Number of parameters: $H^2$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$.
* Multi-head encoder-decoder attention layer using the output of self-attention on the target sequence as queries and the memory sequence as keys and values. Number of parameters: $3 A H (H/A + 1) = 3(H^2 + AH)$.
* Concatenation then projection. One matrix of size $H \times H$. Number of parameters: $H^2$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$.
* Position-wise fully connected layer to hidden layer of size $FH$. One matrix of size $H \times (FH+1)$. Number of parameters: $H(FH+1)$. 
* Position-wise fully connected layer back to output of size $H$. One matrix of size $FH \times (H+1)$. Number of parameters: $FH(H+1)$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$. 
Total number of parameters per residual block:
$$2(3(H^2 + AH) + H^2 + 2H) + H(FH+1) + FH(H+1) + 2H$$
$$=(6 + 2 + F + F)H^2 + (6A + 4 + 1 + F + 2)H$$
$$=(8+2F)H^2 + (6A+ F + 7)H$$

In decoder-only models, the second set of encoder-decoder attention layers is skipped. A decoder-only decoder block consists of:
* Multi-head self-attention layer on the target sequence, using an attention mask to prevent using future information. $3A$ matrices of size $H \times (H/A+1)$. Number of parameters: $3 A H (H/A + 1) = 3(H^2 + AH)$.
* Concatenation then projection. One matrix of size $H \times H$. Number of parameters: $H^2$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$.
* Position-wise fully connected layer to hidden layer of size $FH$. One matrix of size $H \times (FH+1)$. Number of parameters: $H(FH+1)$. 
* Position-wise fully connected layer back to output of size $H$. One matrix of size $FH \times (H+1)$. Number of parameters: $FH(H+1)$. 
* Residual connection.
* Layer norm. Number of parameters: $2H$. 
Total number of parameters per residual block:
$$3(H^2 + AH) + H^2 + 2H + H(FH+1) + FH(H+1) + 2H$$
$$= (3 + 1 + F + F)H^2 + (3A + 2 + 1 + F + 2)H$$
$$= (4+2F)H^2 + (3A + F + 5)H$$

### Word embedding
The word embedding is a look-up table of length $H$ vectors for each of the $V$ words in the vocabulary. It has $HV$ parameters.

## Positional encoding
The positional encoding is a vector of length $H$ representing the location of a word, computed as 
$$PE_{t,2i} = \sin(t/10000^{2i/H})$$
$$PE_{t,2i+1} = \cos(t/10000^{2i/H})$$

The input to the first layer of the encoder/decoder blocks is the sum of the word embedding and the position encoding. 

### Generator
The generator is a linear projection from the output of size $H$ to the vocabulary size $V$ followed by a soft-max and has $HV$ parameters, shared with the  word embedding. This output is interpreted as the probability of outputting each token. 

## Architecture Comparison

### Encoder-Decoder models

**Attention Is All You Need** is an encoder-decoder network with $L = 6$ encoder and decoder blocks, hidden size $H = 512$, number of attention heads is $A=8$, fully-connected factor is $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H] + L[(8+2F)H^2 + (6A+ F + 7)H]$ = 44 million parameters. 

### Encoder-only models

**BERT** is an encoder-only model. 

*BERT-Base* has $L = 12$ encoder blocks, hidden size $H = 768$, $A = 12$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 85 million parameters. 

*BERT-large* has $L = 24$ encoder blocks, hidden size $H = 1024$, $A = 16$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 303 million parameters. 

### Decoder-only models

Decoder-only models were proposed in [Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf). These tend to do conditional text generation, with a prompt input to the network first. 

**T-D** used  $L = 6$ decoder-only blocks, hidden size $H = 512$, number of attention heads is $A=8$, fully-connected factor is $F=4$. Total number of parameters is:
$$L [(4+2F)H^2 + (3A + F + 5)H] = 19 million parameters. 

**GPT-2** is a scaled-up decoder-only model.

*GPT-2-Small* has $L = 12$ decoder-only blocks, hidden size $H = 768$, $A = 12$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 85 million parameters (same as BERT-Base)

*GPT-2-Medium* has $L = 24$ decoder-only blocks, hidden size $H = 1024$, $A = 16$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 303 million parameters (same as BERT-Large)

*GPT-2-Large* has $L = 36$ decoder-only blocks, hidden size $H = 1280$, $A = 20$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 711 million parameters.

*GPT-2-Extra-Large* has $L = 48$ decoder-only blocks, hidden size $H = 1600$, $A = 25$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 1.481 billion parameters. 

**GPT-3**  has $L = 96$ decoder-only blocks, hidden size $H = 12288$, $A = 96$ attention heads, fully-connected factor $F=4$. It also uses $d_{head} = 128$, while all other models use $d_head = 64$. The number of parameters is 175 billion. 

## Optimization

**RoBERTa**: BERT + optimization tweaks, bigger training data set

## References

[Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf)
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
[The Annotated Tranformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

