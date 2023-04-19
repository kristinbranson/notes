# Notes on papers extending LLM approaches to non-language domains

## GATO: [A Generalist Agent](https://arxiv.org/pdf/2205.06175v3.pdf)

This work trains an LLM on many modalities of data by flattening. Each wordpiece, image patch, continuous sensory input, or output action becomes a token, so all observations and outputs at a given time point become a subsequence of the time series. 

I'm incredibly confused about how/if the tokens for one modality are separated from another. In some cases, it appears that they are, and in others it seems like they aren't? It's unclear whether the choices made were arbitrary, or they tried multiple things and this worked best, as the experiments are  not comparative about these details. It does not seem like there is enough information to really understand what was done, but there is an attempt to implement this here: https://github.com/OrigamiDream/gato (Tensorflow). 

