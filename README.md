### Decomposable Attention for Natural Language Inference

This repo implements "A Decomposable Attention Model for Natural Language Inference". You can see the original paper [here](https://arxiv.org/abs/1606.01933).

Before you can run the code in this repo, you need to download the SNLI data from [here](http://nlp.stanford.edu/projects/snli/). Additionally, you will need to alter run.sh to include the data_path argument to where you have saved the data. 

This code uses one of the daily binary tensorflow releases, which you can download [here](https://github.com/tensorflow/tensorflow). There's no reason why it won't work with the standard release, you might just have to do a bit of refactoring.

The model offers a debug flag which runs using only a few examples(running without this flag will require around 10G of RAM or for you to change the number of buckets that we load - I have been using around 7). The general approach here is to generate pre-batched, pre-padded buckets of different lengthed premises and hypothesis and then to generate a Tensorflow model per bucket, which all share parameters.

Additionally, there is a multi-threading option to perform simultaneous single epoch steps. 

##### Note: I have found that I can't get this implementation past ~83.6% Validation accuracy. If you spot a bug/inconsistancy please let me know! 
