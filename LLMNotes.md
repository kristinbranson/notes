
# Large Language Model Notes

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

### AIAYN Decoder blocks
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

### Decoder-only blocks

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

Decoder-only and encoder-only models differ only in their attention masks, as far as I can tell.

### Word embedding
The word embedding is a look-up table of length $H$ vectors for each of the $V$ words in the vocabulary. It has $HV$ parameters.

## Positional encoding
The positional encoding is a vector of length $H$ representing the location of a word, computed as 
$$PE_{t,2i} = \sin(t/10000^{2i/H})$$
$$PE_{t,2i+1} = \cos(t/10000^{2i/H})$$

The input to the first layer of the encoder/decoder blocks is the sum of the word embedding and the position encoding. 

### Generator
The generator is a linear projection from the output of size $H$ to the vocabulary size $V$ followed by a soft-max and has $HV$ parameters, shared with the  word embedding. This output is interpreted as the probability of outputting each token. 

### Memory

Transformer XL introduced a recurrence mechanism.  During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment, thus having access to 2X the context without additional training compute. The one wrinkle here is the cached positional encoding is incorrect. To fix this, they propose replacing the positional encoding with position-related terms in each attention layer.
Originally:
$$a_{ij}' = (z_i+p_i)^\top W_q^\top W_k (x_j+p_j) = z_i^\top W_q^\top W_k x_j + z_i^\top W_q^\top W_k p_j + p_i^\top W_q^\top W_k x_j + p_i^\top W_q^\top W_k p_j$$ 
where $p_j$ is the positional encoding. They propose
$$a_{ij}' = z_i^\top W_q^\top W_{Xk} x_j + z_i^\top W_q^\top W_{Rk} R_{i-j} + u^\top W_{Xk} x_j + v^\top W_{Rk} R_{i-j}$$ 
where $R_{i-j}$ is a sinusoidal relative positional encoding and $u$ and $v$ are learned vectors, and two separate key matrices $W_{Xk}$ and $W_{Rk}$ are learned. 

## Architecture Comparison

### Encoder-Decoder models

**Attention Is All You Need** is an encoder-decoder network with $L = 6$ encoder and decoder blocks, hidden size $H = 512$, number of attention heads is $A=8$, fully-connected factor is $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H] + L[(8+2F)H^2 + (6A+ F + 7)H]$ = 44 million parameters. 

### Encoder-only models

**BERT** is an encoder-only model trained with a masked loss. 

*BERT-Base* has $L = 12$ encoder blocks, hidden size $H = 768$, $A = 12$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 85 million parameters. 

*BERT-large* has $L = 24$ encoder blocks, hidden size $H = 1024$, $A = 16$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 303 million parameters. 

**ELECTRA** discriminator has the same model architecture as BERT-Base. The generator used for corrupting has a smaller width layers. 

### Decoder-only models

Decoder-only models were proposed in [Generating Wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf). These tend to do conditional text generation, with a prompt input to the network first. 

**T-D** is trained with a causal loss. It used  $L = 6$ decoder-only blocks, hidden size $H = 512$, number of attention heads is $A=8$, fully-connected factor is $F=4$. Total number of parameters is:
$$L [(4+2F)H^2 + (3A + F + 5)H] = 19 million parameters. 

**GPT-X** is a scaled-up T-D model. 

*GPT* = *GPT-2-Small* has $L = 12$ decoder-only blocks, hidden size $H = 768$, $A = 12$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 85 million parameters (same as BERT-Base)

*GPT-2-Medium* has $L = 24$ decoder-only blocks, hidden size $H = 1024$, $A = 16$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 303 million parameters (same as BERT-Large)

*GPT-2-Large* has $L = 36$ decoder-only blocks, hidden size $H = 1280$, $A = 20$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 711 million parameters.

*GPT-2-Extra-Large* has $L = 48$ decoder-only blocks, hidden size $H = 1600$, $A = 25$ attention heads, fully-connected factor $F=4$. The number of parameters, excluding the embedding and generator layers, is 
$L [(4+2F)H^2 + (3A + F + 5)H]$ = 1.481 billion parameters. 

**GPT-3**  has $L = 96$ decoder-only blocks, hidden size $H = 12288$, $A = 96$ attention heads, fully-connected factor $F=4$. It also uses $d_{head} = 128$, while all other models use $d_head = 64$. The number of parameters is 175 billion. 

**XLNet** is a decoder-only model trained with a permutation loss. 

## Prompting vs Fine-Tuning

Most models suggest self-supervised training on large unlabeled text data sets, followed by training heads for specific tasks. 

GPT-2 and GPT-3 suggest instead doing self-supervised training only, and encoding the task through a prompt. 

## Hyperparameters

Amount of training data, batch size, and training iterations are growing as model size grows. 

**RoBERTa**: BERT + optimization tweaks, bigger training data set. 

* Attention is all you need: 300 thousand iterations, batch size 25000 source and target tokens, 36M sentences training data (English-French)
* BERT: 1 million iterations, batch size = 256, 16GB training data
* RoBERTa: 500 thousand iterations, batch size = 8K, 160GB training data
* XLNet: 500 thousand iterations, batch size = 8K, 126GB = 33B tokens training data
* GPT: 100 epochs, batch size = 64, ?? training data, 512 tokens
* GPT-2: ?? iterations, batch size = 512, ?? training data, 1024 tokens context
* GPT-3: trained for 300B tokens, batch size = 3.2M, 500B tokens training data, 1024 tokens context 
* ELECTRA: 1 million iterations, batch size = 256, training data same as XLNet: 126GB = 33B tokens

[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf) empirically shows rules for choosing model size, batch size, and training iterations given a compute budget (for T-D-style models, context = 1024 tokens). If $C$ is your compute budget:
* Number of parameters: $N \propto C^{\alpha_C^{min}/\alpha_N}$
* Batch size: $B \propto C^{\alpha_C^{min}/\alpha_B}$. Decreasing batch size does not worsen compute requirements, but, if parallelization across multiple GPUs is possible, larger batch sizes allow more parallelization. 
* Iteration steps: $S \propto C^{\alpha_C^{min}/\alpha_S}
* Data set size $D = BS$

where
* $\alpha_C^{min} = (\alpha_S^{-1}+\alpha_B^{-1}+\alpha_N^{-1})^{-1}$
* $\alpha_S = .76$
* $\alpha_B = .21$
* $\alpha_N = .076$

They found that results at convergence were largely independent of learning rate schedule. Unless otherwise noted, they used a learning rate schedule with a 3000 step linear warmup followed by a cosine decay to zero.

Model shape didn't matter much -- little effect of changing $F$, $H$, and $L$ when total number of parameters was kept constant. $F$ should be around $1-2$, $H/L$ should be $25-100$, $H/A$ should be $12-100$. 

[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) got slightly different results. While OpenAI's paper suggested for each 10X increase in compute you had, you should increase model size by $5x$ and data size by $2x$, DeepMind found that you should increase each equally ($\sqrt{10}x$). 

## Tokenization

## Training Data

## Benchmarks

Types of tasks:
1. Are two sentences semantically equivalent (or riffs on this -- does one follow the other, do they contradict, neutral, is the second an answer to the first)? 
2. Is this a linguistically acceptable sentence? 
3. Sentiment prediction, e.g. is this a positive review? 
4. Find the span of text from a passage that answers a question. 

### GLUE: General Language Understanding Evaluation
From [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) appendix:
* **MNLI** (1) Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task. Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.
* **QQP** (1) Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent
* **QNLI** (1) Question Natural Language Inference is a version of the Stanford Question Answering Dataset which has been converted to a binary classification task. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.
* **SST-2** (3) The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment. 
* **CoLA** (2) The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not. 
* **STS-B** (1) The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources. They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning. 
* **MRPC** (1) Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent. 
* **RTE** (1) Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data. 

### SQuAD: Stanford Question Answering Dataset

SQuAD v1.1 (4) is 100k crowdsourced question/answer pairs. Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage. SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.

### SWAG: Situations With Adversarial Generations

SWAG (1) contains 113k sentence-pair completion examples that evaluate grounded common-sense inference. Given a sentence, the task is to choose the most plausible continuation among four choices. 

### RACE: Large-scale ReAding Comprehension Dataset From Examinations

RACE (1) contains 28K passages and 100K questions taken from the English exams for middle and high school Chinese students in the age range between 12 to 18, with multiple choice answers generated by human experts. Answers are not restricted to be text spans in the original passage. 

### IMDB 

IMDB (3) dataset has 50K highly polar movie reviews, and the goal is to predict whether the review is positive or negative. 

### Yelp

Yelp dataset has 1,569,264 reviews with the goal of predicting the rating (stars) or polarity.

### DBPedia ontology dataset

DBPedia (3) consists of 45,000 wikipedia articles from 14 non-overlaping topics, with the goal of predicting the topic from the text. 

### AG

AG (3) is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. The goal is to predict document topic.

### Amazon

Amazon (3) consists of 34,686,770 reviews from 6,643,669 users on 2,441,053 products over 18 years, with the goal of predicting rating or polarity. 

## Fine-tuning

BERT adds an additional output layer for fine-tuning. It appears that all parameters are fine-tuned, not just the last layer. For sentence-pair classification tasks, it encodes both sentences together, separated with the [SEP] token. For SQuAD, the question and passage are separated by the [SEP] token. They predict whether each token is a start or end of a span, and choose the span with the highest score. For SQuAD 2.0, they set the start and end to be the [CLS] token. For SWAG, they construct 4 sentence pair inputs, one for each possible continuation. For text generation, I believe that MLMs have a sequence of [MASK] tokens in front of them, and iteratively predict the next token. 

## References

1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
3. [The Annotated Tranformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
4. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
5. [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf)
6. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
7. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
8. [ELECTRA: Pre-Training Text Encoders as Discriminators Rather than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
9. [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)
10. [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
11. [LessWrong: New Scaling Laws for Large Language Models](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models)
