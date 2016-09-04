import time
import numpy as np
from random import shuffle
from itertools import chain
import tensorflow as tf
import threading
from queue import Queue



def bucket_shuffle(dict_data):
    # zip each data tuple with it's bucket id.
    # return as a randomly shuffled iterator.
    id_to_data =[]
    for x, data in dict_data.items():
        id_to_data += list(zip([x]*len(data), data))

    shuffle(id_to_data)

    return len(id_to_data), iter(id_to_data)


def run_epoch(session, models, data, training, verbose=False):
    """Runs the model on the given data."""
    #epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    accuracy = 0.0
    first_acc = 0.0
    epoch_size, id_to_data = bucket_shuffle(data)

    for step in range(epoch_size):
        (id,(x, y)) = next(id_to_data)
        m = models[id]
        assert x["premise"].shape == (m.premise.get_shape())
        assert x["hypothesis"].shape == (m.hypothesis.get_shape())

        if training:
            eval_op = m.train_op

            batch_acc, cost, _ = session.run([m.accuracy, m.cost, eval_op], feed_dict={m.premise: x["premise"],
                                      m.hypothesis: x["hypothesis"],
                                      m.targets: y})
        else:

            batch_acc, cost  = session.run([m.accuracy, m.cost], feed_dict={m.premise: x["premise"],
                                      m.hypothesis: x["hypothesis"],
                                      m.targets: y})

        costs += cost
        iters += 1
        accuracy += batch_acc
        #if step % (epoch_size // 10) == 10:
        print("%.3f acc: %.3f loss: %.3f speed: %.0f examples/s" %
              (step * 1.0 / epoch_size,
               accuracy / iters,
               costs / iters,
               iters * m.batch_size / (time.time() - start_time)))


    return (costs / iters), (accuracy / iters)

def async_single_epoch(num_threads, session, models, data, verbose=False):

    def single_thread_epoch(coordinator, session):

        epoch_size, id_to_data = bucket_shuffle(data)

        step = 0.0
        start_time = time.time()
        accuracy = 0.0
        while not coordinator.should_stop():
            try:
                (id,(x, y)) = next(id_to_data)
                m = models[id]
                step +=1

                acc, _ = session.run([m.accuracy, m.train_op], feed_dict={m.premise: x["premise"],
                                      m.hypothesis: x["hypothesis"],
                                      m.targets: y})
                accuracy += acc
                print("{} acc: {} speed: {} examples/s".format(step * 1.0 / epoch_size,
                              accuracy / step,
                              step * m.batch_size / (time.time() - start_time)))
            except:
                print("Requesting stop")
                coordinator.request_stop()

    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=single_thread_epoch, args=(coord,session)) for x in range(num_threads)]

    # Start the threads and wait for all of them to stop.
    print("Starting {} optimisation threads".format(num_threads))
    for t in threads: t.start()
    coord.join(threads)

