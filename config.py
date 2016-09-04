from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DAConfig(object):

  init_scale = 0.01
  learning_rate = 0.05
  max_grad_norm = 5
  hidden_size = 200
  max_max_epoch = 16
  keep_prob = 0.8
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000 # overriden by the actual vocab size you generate

  embedding_size = 300
  embedding_reg = None
  train_embeddings = False
  use_embeddings = True




