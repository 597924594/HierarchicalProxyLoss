# HierarchicalProxyLoss

## Description

This is the repo for our Diginal Signal Processing Paper: Hierarchical Multiple Proxy Loss for Deep Metric Learning.

## Installation
#### Environment
* python=3.6
* pytorch=1.7.1
* [powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning)

#### Dataset
Datasets will be download and installed automaticly as long as set `download: True` in the training command line.

## Train and Test
Please refer to the `scripts.sh` for the command line to train and test models on three benchmarker datasets using the default hyper-parameters for the proposed method. 

The evaluation score will be stored in `[{experiment_name}/meta_logs/saved_csvs/ConcatenateEmbeddings_accuracies_normalized_compared_to_self_GlobalEmbeddingSpaceTester_level_0_TEST.csv]`. Please refer to [powerful_benchmarker](https://kevinmusgrave.github.io/powerful-benchmarker/#experiment-folder-format) for more details.

## Quantify results
The results have been reported in our paper. The training checkpoint can be downloaded from 