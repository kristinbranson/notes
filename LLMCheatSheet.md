
# LLM Cheat Sheet

## Pre-training loss functions

**Causal** language models: predict next word given previous words. Training loss is on all words in the block, attention masks are used so that Used in Attention is All You Need, GPT-x. 

**Masked** language models: Replace words with token [MASK], try to predict word using all words around that word. Used in BERT. 

**Permutation** language models: construct attention masks based on random permutations, predict words from randomly selected words around the word. Used in XLNet. 

**Replaced token detection**:  replace tokens with words generated from small generator network, predict whether tokens are replaced or not. Used in ELECTRA. 

## Architectures

Encoder blocks: These consist of self-attention layers on the input:
$$h_{ij} = x_i^\top W_q^\top W_k x_j W_v x_j$$
See [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) for a good visualization, [The Annotated Tranformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) for sample code. 

**Encoder-decoder**: N encoding transformer blocks feeding into N decoding transfomer blocks. 

## Optimization

**RoBERTa**: BERT + optimization tweaks, bigger training data set
