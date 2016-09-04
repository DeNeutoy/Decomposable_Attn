

import tensorflow as tf
from tensorflow.python.ops import  rnn, rnn_cell, seq2seq
from embedding_utils import input_projection3D

class DAModel(object):

    """ Implements a Decomposable Attention model for NLI.
        http://arxiv.org/pdf/1606.01933v1.pdf """

    def __init__(self, config, pretrained_embeddings=None,update_embeddings=True, is_training=False):

        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        self.vocab_size = config.vocab_size
        self.prem_steps = config.prem_steps
        self.hyp_steps = config.hyp_steps
        self.is_training = is_training
        # placeholders for inputs
        self.premise = tf.placeholder(tf.int32, [batch_size, self.prem_steps])
        self.hypothesis = tf.placeholder(tf.int32, [batch_size, self.hyp_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, 3])

        if pretrained_embeddings is not None:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_size], dtype=tf.float32,
                                        trainable=update_embeddings)
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.config.embedding_size])
            self.embedding_init = embedding.assign(self.embedding_placeholder)
        else:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.hidden_size], dtype=tf.float32)


        # create lists of (batch,step,hidden_size) inputs for models
        premise_inputs = tf.nn.embedding_lookup(embedding, self.premise)
        hypothesis_inputs = tf.nn.embedding_lookup(embedding, self.hypothesis)


        if pretrained_embeddings is not None:
            with tf.variable_scope("input_projection"):
                premise_inputs = input_projection3D(premise_inputs, self.hidden_size)
            with tf.variable_scope("input_projection", reuse=True):
                hypothesis_inputs = input_projection3D(hypothesis_inputs, self.hidden_size)

        # run FF networks over inputs
        with tf.variable_scope("FF"):
            prem_attn = self.feed_forward_attention(premise_inputs)
        with tf.variable_scope("FF", reuse=True):
            hyp_attn = self.feed_forward_attention(hypothesis_inputs)

        # This is doing all the dot-products for the feedforward attention at once.
        # get activations, shape: (batch, prem_steps, hyp_steps )
        dot = tf.batch_matmul(prem_attn, hyp_attn, adj_y=True)

        hypothesis_softmax = tf.reshape(dot, [batch_size*self.prem_steps, -1,]) #(300,10)
        hypothesis_softmax = tf.expand_dims(tf.nn.softmax(hypothesis_softmax),2)

        dot = tf.transpose(dot, [0,2,1]) # switch dimensions so we don't screw the reshape up

        premise_softmax = tf.reshape(dot, [batch_size*self.hyp_steps, -1]) #(200,15)
        premise_softmax = tf.expand_dims(tf.nn.softmax(premise_softmax),2)

        # this is very ugly: we make a copy of the original input for each of the steps
        # in the opposite sentence, multiply with softmax weights, sum and reshape.
        alphas = tf.reduce_sum(premise_softmax *
                               tf.tile(premise_inputs, [self.hyp_steps, 1, 1]), [1])
        betas = tf.reduce_sum(hypothesis_softmax *
                              tf.tile(hypothesis_inputs, [self.prem_steps, 1 ,1 ]), [1])

        # this is a list of (batch, hidden dim) tensors of hyp_steps length
        alphas = [tf.squeeze(x) for x in
                  tf.split(1,self.hyp_steps,tf.reshape(alphas, [batch_size, -1, self.hidden_size]))]
        # this is a list of (batch, hidden dim) tensors of prem_steps length
        betas = [tf.squeeze(x) for x in
                 tf.split(1, self.prem_steps,tf.reshape(betas, [batch_size, -1 , self.hidden_size]))]

        # list of original premise vecs to go with betas
        prem_list = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.prem_steps, premise_inputs)]

        # list of original hypothesis vecs to go with alphas
        hyp_list = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.hyp_steps, hypothesis_inputs)]

        beta_concat_prems = []
        alpha_concat_hyps = []

        # append the relevant alpha/beta to the original word representation
        for input, rep in zip(prem_list, betas):
            beta_concat_prems.append(tf.concat(1,[input,rep]))

        for input, rep in zip(hyp_list, alphas):
            alpha_concat_hyps.append(tf.concat(1, [input, rep]))


        # send both through a feedforward network with shared parameters
        with tf.variable_scope("compare"):
            prem_comparison_vecs = tf.split(0,self.prem_steps,
                    self.feedforward_network(tf.concat(0, beta_concat_prems)))

        with tf.variable_scope("compare", reuse=True):
            hyp_comparison_vecs = tf.split(0,self.hyp_steps,
                    self.feedforward_network(tf.concat(0, alpha_concat_hyps)))

        # add representations and send through last classifier
        sum_prem_vec = tf.add_n(prem_comparison_vecs)
        sum_hyp_vec = tf.add_n(hyp_comparison_vecs)

        with tf.variable_scope("final_representation"):
            final_representation = self.feedforward_network(tf.concat(1, [sum_prem_vec, sum_hyp_vec]))

        # softmax over outputs to generate distribution over [neutral, entailment, contradiction]
        softmax_w = tf.get_variable("softmax_w", [4*hidden_size, 3])
        softmax_b = tf.get_variable("softmax_b", [3])
        self.logits = tf.matmul(final_representation, softmax_w) + softmax_b   # dim (batch_size, 3)

        _, targets = tf.nn.top_k(self.targets)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([batch_size])],
                3)
        self.cost = tf.reduce_mean(loss)

        _, logit_max_index = tf.nn.top_k(self.logits)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(logit_max_index, targets), tf.float32))


        if is_training:

            self.lr = tf.Variable(self.config.learning_rate, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.config.max_grad_norm)

            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def feed_forward_attention(self, attendees):
        "Sends 3D tensor through two 2D convolutions with 2 different features"

        attn_length = attendees.get_shape()[1].value
        attn_size = attendees.get_shape()[2].value

        hidden = tf.reshape(attendees, [-1, attn_length, 1, attn_size])
        k1 = tf.get_variable("W1", [1,1,attn_size,attn_size])
        k2 = tf.get_variable("W2", [1,1,attn_size,attn_size])
        b1 = tf.get_variable("b1", [attn_size])
        b2 = tf.get_variable("b2", [attn_size])

        if self.config.keep_prob < 1.0 and self.is_training:
            hidden = tf.nn.dropout(hidden, self.config.keep_prob)

        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hidden, k1, [1, 1, 1, 1], "SAME"),b1))

        if self.config.keep_prob < 1.0 and self.is_training:
            features = tf.nn.dropout(features, self.config.keep_prob)

        features = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(features, k2, [1,1,1,1], "SAME"), b2))

        return tf.squeeze(features)



    def feedforward_network(self, input):

        hidden_dim = input.get_shape()[1].value

        hidden1_w = tf.get_variable("hidden1_w", [hidden_dim, hidden_dim])
        hidden1_b = tf.get_variable("hidden1_b", [hidden_dim])

        hidden2_w = tf.get_variable("hidden2_w", [hidden_dim, hidden_dim])
        hidden2_b = tf.get_variable("hidden2_b", [hidden_dim])

        if self.config.keep_prob < 1.0 and self.is_training:
            input = tf.nn.dropout(input, self.config.keep_prob)

        hidden1 = tf.nn.relu(tf.matmul(input, hidden1_w) + hidden1_b)

        if self.config.keep_prob < 1.0 and self.is_training:
            hidden1 = tf.nn.dropout(hidden1, self.config.keep_prob)

        gate_output = tf.nn.relu(tf.matmul(hidden1, hidden2_w) + hidden2_b)

        return gate_output

