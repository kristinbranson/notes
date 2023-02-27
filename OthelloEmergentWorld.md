# Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task

In our reading group last week, we discussed **Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task** (Li et al., ICLR 2023, https://arxiv.org/abs/2210.13382). As large language models (LLMs) are only trained to predict the next word in a sequence, the authors ask whether they only memorize surface statistics, or if they learn a model of the world that generates those words. This is an important question in understanding what LLMs can do and how they do it.

They do this by studying a simple world, the board game Othello. Representing Othello board locations as word tokens, they train a GPT-like model to predict the next tile a piece is played to. The learning algorithm is told nothing about the rules of Othello or, importantly, the structure of the board. It is just given sequences like [44, 45, 37, 43, 34, 20, 19, 38, 21, 33, 42, 51, 41, 25] and trained to predict 52, a legal move (0 is the top-left corner, 59 is the bottom-right corner tile).

<img src="https://user-images.githubusercontent.com/211380/221597133-f7e62c9b-630f-4397-b3fe-4e7207e5adc6.png" alt="Othello board with token numbers" style="width:300px;"/>

They then ask whether the state of the board (which color tile is in each location) is stored in the hidden state learned by the transformer by training a two-layer network to predict the state board from the hidden representation, and show that they can do this accurately (best error is in layer 7 out of 8). A linear classifier failed. 

![image](https://user-images.githubusercontent.com/211380/221600842-605438ca-d539-49f9-986c-303b15b1ff69.png)

They also show that they can manipulate the board state in the internal representation, resulting in model predictions corresponding to the new board state. Because transformers input a sequence, the necessary intervention was non-standard and somewhat complex -- they had to nudge the internal representation at ~5 layers in sequence. As we develop XAI methods for transformers, this is something to consider. 

![image](https://user-images.githubusercontent.com/211380/221597429-dc2f0262-bf6b-4f5d-924e-795ee809045b.png)
![image](https://user-images.githubusercontent.com/211380/221600917-83dc2aaa-e912-4deb-a183-dfee62204a1e.png)

The paper is convincing in showing that the world of Othello -- the board state -- is learned by the LLM. Some questions I have.
1. I wonder how surprising this should be. What are the models that can predict legal Othello moves but do not represent board state, besides a lookup table? We can think of self-supervised next-word training as a sort of autoencoder, and we expect autoencoders to learn something nontrivial about the structure of our data. Is there an advantage to unsupervised learning as done by an LLM over more classical techniques (I think so!)? 
2. Does this tell us anything about whether LLMs trained on internet text learn about the real world? There's a strict mapping between syntax and semantics here which may not be the case in some types of text. Could we test whether there is a world model learned for math word problems? 

Slides from my presentation:
https://docs.google.com/presentation/d/1KcK6qisPXcs-ZYVs1x6QKM-IPoAfTLZgDZnt4y00diE/edit?usp=sharing
