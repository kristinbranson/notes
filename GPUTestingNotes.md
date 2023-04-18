# Notes on testing a GPU

My computer started crashing during training, so I've been running some tests using the GPU to see if I can replicate it outside of my code. I had this issue in the past and I solved it by upgrading pytorch. I don't know. In any case, here are some notes on how to run various GPU tests:

## Lambda Benchmarks

Lambda benchmarks train the following:
* [PyTorch_ncf_FP32](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/NCF/README.md)
* [PyTorch_tacotron2_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_resnet50_AMP](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) (synthetic data backend)
* [PyTorch_bert_large_squad_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
* [PyTorch_ncf_FP16](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Recommendation/NCF/README.md)
* [PyTorch_waveglow_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_transformerxllarge_FP32](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/README.md)
* [PyTorch_gnmt_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)
* [PyTorch_bert_large_squad_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
* [PyTorch_SSD_AMP](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
* [PyTorch_transformerxlbase_FP16](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/README.md)
* [PyTorch_bert_base_squad_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
* [PyTorch_transformerxllarge_FP16](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/README.md)
* [PyTorch_tacotron2_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_waveglow_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
* [PyTorch_SSD_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
* [PyTorch_resnet50_FP32](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) (synthetic data backend)
* 

### Set up

Followed instructions here:
https://github.com/lambdal/deeplearning-benchmark/blob/master/pytorch/README.md

1. Cloned the repo at https://github.com/lambdal/deeplearning-benchmark into ~/software/deeplearning-benchmark:
```
# Lambda's fork of DeepLearningExamples (a few patches to make sure they work with the recent NGC)
git clone https://github.com/LambdaLabsML/DeepLearningExamples.git && \
cd DeepLearningExamples && \
git checkout lambda/benchmark && \
cd ..

# Clone this repo for streamlining the benchmark
git clone https://github.com/lambdal/deeplearning-benchmark.git && \
cd deeplearning-benchmark/pytorch
```
2. Pulled the NGC (NVIDIA GPU Cloud) docker container for the latest PyTorch:
```
export NAME_NGC=pytorch:22.10-py3
docker pull nvcr.io/nvidia/${NAME_NGC}
```
3. Prepared data
```
```
4. Created config file scripts/config_v1/config_pytorch_titan_rtx_24GB.sh for my Titan RTX with 24GB RAM. 
5. Ran benchmark:
```
```
