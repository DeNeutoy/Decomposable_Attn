### Decomposable Attention for Natural Language Inference

This repo implements "A Decomposable Attention Model for Natural Language Inference". You can see the original paper [here](https://arxiv.org/abs/1606.01933).

Before you can run the code in this repo, you need to download the SNLI data from [here](http://nlp.stanford.edu/projects/snli/). Additionally, you will need to alter run.sh to include the data_path argument to where you have saved the data. 

The model offers a debug flag which runs using only a few examples(running without this flag will require around 10G of RAM or for you to change the number of buckets that we load - I have been using around 7). The general approach here is to generate pre-batched, pre-padded buckets of different lengthed premises and hypothesis and then to generate a Tensorflow model per bucket, which all share parameters.

Additionally, there is a multi-threading option to perform simultaneous single epoch steps. 
