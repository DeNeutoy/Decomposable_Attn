import csv
import tensorflow as tf

def import_embeddings(filename):
    words = {}
    with open(filename, 'rt', encoding='utf8') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for row in r:
            words[row[0]] = row[1:301]
    return words

def input_projection3D(input3D, projection_size):

    hidden = input3D.get_shape()[2].value
    steps = input3D.get_shape()[1].value
    if hidden < projection_size : print("WARNING - projecting to higher dimension than original embeddings")
    inputs = tf.reshape(input3D, [-1, steps, 1, hidden]) # now shape (batch, num_steps, 1, hidden_size)
    W_proj = tf.get_variable("W_proj", [1,1,hidden, projection_size])
    b_proj = tf.get_variable("b_proj", [projection_size])

    projection = tf.nn.conv2d(inputs, W_proj, [1,1,1,1], "SAME")
    projection = tf.tanh(tf.nn.bias_add(projection,b_proj))
    return tf.reshape(projection, [-1, steps,projection_size])