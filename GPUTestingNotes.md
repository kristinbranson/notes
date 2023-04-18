# Notes on testing a GPU

My computer started crashing during training, so I've been running some tests using the GPU to see if I can replicate it outside of my code. I had this issue in the past and I solved it by upgrading pytorch. I don't know. In any case, here are some notes on how to run various GPU tests:

## Lambda Benchmarks

Lambda benchmarks train the following:
* [PyTorch_ncf_FP32](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/NCF/README.md)
* [PyTorch_tacotron2_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_resnet50_AMP](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)
* [PyTorch_bert_large_squad_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
* [PyTorch_ncf_FP16](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/NCF/README.md)
* [PyTorch_waveglow_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_transformerxllarge_FP32](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/README.md)
* [PyTorch_gnmt_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)
