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
* [PyTorch_transformerxlbase_FP32](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/README.md)
* [PyTorch_gnmt_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/GNMT)
* [PyTorch_bert_base_squad_FP16](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)

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
3. Prepared data. In directory ~/software/deeplearning-benchmark/pytorch:
```
docker run --gpus all --rm --shm-size=64g -v ~/software/DeepLearningExamples/PyTorch:/workspace/benchmark -v ~/software/deeplearning-benchmark_data:/data -v $(pwd)"/scripts":/scripts nvcr.io/nvidia/${NAME_NGC} /bin/bash -c "cp -r /scripts/* /workspace;  ./run_prepare.sh"
```
4. Created config file scripts/config_v1/config_pytorch_titan_rtx_24GB.sh for my Titan RTX with 24GB RAM. 
5. Ran benchmark. In directory ~/software/deeplearning-benchmark/pytorch:
```
docker run --gpus all --rm --shm-size=64g -v ~/software/DeepLearningExamples/PyTorch:/workspace/benchmark -v ~/software/deeplearning-benchmark_data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} /bin/bash -c "cp -r /scripts/* /workspace;  ./run_benchmark.sh titan_rtx_24GB all 3000"
```
6. Gather results. Modified the script scripts/compile_results_pytorch.py to include
` "titan_rtx_24GB": ([1, 1], "KB Titan RTX", 280, 2500),` under `list_system_single`. Also modified the names and paths of the output files to 
```
df_throughput.to_csv("/results/pytorch-train-throughput-" + args.precision + "-kb.csv")
...
df_bs.to_csv("/results/pytorch-train-bs-" + args.precision + "-kb.csv")
```
Ran: 
```
docker run --gpus all --rm --shm-size=64g -v ~/software/DeepLearningExamples/PyTorch:/workspace/benchmark -v ~/software/deeplearning-benchmark_data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} /bin/bash -c "cp -r /scripts/* /workspace;  python compile_results_pytorch.py --precision fp32 --system all --path /results"
```
and
```
docker run --gpus all --rm --shm-size=64g -v ~/software/DeepLearningExamples/PyTorch:/workspace/benchmark -v ~/software/deeplearning-benchmark_data:/data -v $(pwd)"/scripts":/scripts -v $(pwd)"/results":/results nvcr.io/nvidia/${NAME_NGC} /bin/bash -c "cp -r /scripts/* /workspace;  python compile_results_pytorch.py --precision fp16 --system all --path /results"
```

### Results

Train throughput (higher is better): 

FP32: 

 | name_gpu            | num_gpu | watt  | price   | ssd   | bert_base_squad | bert_large_squad | gnmt     | ncf        | resnet50 | tacotron2 | transformerxlbase | transformerxllarge | waveglow |
 | ------------------- | ------- | ----- | ------- | ----- | --------------- | ---------------- | -------- | ---------- | -------- | --------- | ----------------- | ------------------ | -------- |
 | KB Titan RTX        | 1.0     | 280.0 | 2500.0  | 88.0  | 45.0            | 13.0             | 23352.0  | 8473545.0  | 301.0    | 17623.0   | 6960.0            | 2311.0             | 50108.0  |
 | Quadro RTX 8000     | 1.0     | 260.0 | 6037.5  | 81.0  | 41.0            | 12.0             | 21616.0  | 8501458.0  | 274.0    | 17717.0   | 6623.0            | 2515.0             | 44068.0  |
 | H100 80GB PCIe Gen5 | 1.0     | 350.0 | 30918.0 | 355.0 | 292.0           | 97.0             | 162431.0 | 32325375.0 | 1197.0   | 61491.0   | 34241.0           | 15459.0            | 262302.0 |

FP16:

| name_gpu            | num_gpu | watt  | price    | ssd   | bert_base_squad | bert_large_squad | gnmt     | ncf        | resnet50 | tacotron2 | transformerxlbase | transformerxllarge | waveglow | 
| ------------------- | ------- | ----- | -------- | ----- | --------------- | ---------------- | -------- | ---------- | -------- | --------- | ----------------- | ------------------ | -------- | 
| KB Titan RTX        | 1.0     | 280.0 | 2500.0   | 197.0 | 159.0           | 53.0             | 85557.0  | 17555125.0 | 695.0    | 17783.0   | 17849.0           | 6339.0             | 50050.0  |
| Quadro RTX 8000     | 1.0     | 260.0 | 6037.5   | 182.0 | 146.0           | 50.0             | 88498.0  | 19801310.0 | 651.0    | 18587.0   | 18223.0           | 7853.0             | 43964.0  |
| H100 80GB PCIe Gen5 | 1.0     | 350.0 | 30918.0  | 782.0 | 812.0           | 301.0            | 467071.0 | 52418903.0 | 2616.0   | 78108.0   | 62259.0           | 30841.0            | 393500.0 |

### Artifacts:

* ~/software/deeplearning-benchmark: Lambda benchmark code
* ~/software/DeepLearningExamples: Lambda's fork of NVIDIA's DeepLearningExamples
* ~/software/deeplearning-benchmark_data: Data for benchmarks
* ~/software/deeplearning-benchmark/results/titan_rtx_24GB/: Output of benchmarking.
* ~/software/deeplearning-benchmark/results/pytorch-train-\*-kb.csv: Gathered results. 

## GPU Burn
Multi-GPU CUDA stress test: http://wili.cc/blog/gpu-burn.html

Cloned repo into ~/software:
```
git clone git@github.com:wilicc/gpu-burn.git
```
Followed instructions in README:
```
cd gpu-burn
docker build -t gpu_burn .
docker run --rm --gpus all gpu_burn /bin/bash -c "./gpu_burn -d 3600"
```
Note that this did **not** work in the head of the repo, I had to go back to the commit I last tried in September to get this to run:
```
git checkout c535b677236586fdb35655cfa75156304a9cb1f6
```
Intermediate output:
```
./gpu_burn -d 3600
Burning for 3600 seconds.
GPU 0: NVIDIA T400 (UUID: GPU-d02e054f-0e6a-da13-bb1b-81deedaa1e8e)
GPU 1: NVIDIA TITAN RTX (UUID: GPU-b7f60d37-616e-9682-6606-7f2c0343c0dc)
Initialized device 0 with 24220 MB of memory (23879 MB available, using 21491 MB of it), using DOUBLES
Results are 33554432 bytes each, thus performing 669 iterations
Initialized device 1 with 1875 MB of memory (725 MB available, using 653 MB of it), using DOUBLES
Results are 33554432 bytes each, thus performing 18 iterations
4.8%  proc'd: 4683 (518 Gflop/s) - 414 (44 Gflop/s)   errors: 0 - 0   temps: 70 C - 67 C 
```
