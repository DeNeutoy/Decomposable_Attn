
import json
import os
import numpy as np
import argparse
from collections import Counter
from Vocab import Vocab
from nltk.tokenize import word_tokenize
from collections import defaultdict
from random import shuffle


def get_sentences(data):
    # tokenise using nltk tokeniser
    str1 = data["sentence1"]
    str2 = data["sentence2"]

    s1 = word_tokenize(str1.lower())
    s2 = word_tokenize(str2.lower())
    return s1, s2

def pad_sentence(token_list, pad_length, pad_id):

    padding = [pad_id] * (pad_length - len(token_list))
    padded_list = padding + token_list
    return padded_list

def categorical_label(data):

    labels =  ["neutral", "entailment", "contradiction"]
    try:
        category = labels.index(data["gold_label"])
        onehot = [0,0,0]
        onehot[category] = 1
        return onehot

    except ValueError:
        return None


def load_data(data_path,train,dev,test, vocab,
              update_vocab,buckets,max_records=None, batch_size=30):

    all_output = []
    all_stats = []
    max_len = buckets[-1] # last bucket size is the max
    for file in [train, dev, test]:

        output = [[] for _ in range(len(buckets))]
        bucket_dict =  {i:defaultdict(list) for i in range(len(buckets))}

        stats = Counter()

        with open(os.path.join(data_path,file), "r") as dataset:

            for line in dataset:
                data = json.loads(line)

                label = categorical_label(data)
                if label is None:
                    stats['bad_label'] += 1
                    continue

                s1, s2 = get_sentences(data)
                len_s1 = len(s1)
                len_s2 = len(s2)
                stats["max_len_premise"] = max(stats["max_len_premise"], len_s1)
                stats["max_len_hypothesis"] = max(stats["max_len_hypothesis"], len_s2)


                # drop item if either premise or hyp is too long
                if max_len and (len_s1 > max_len[0] or len_s2 > max_len[1]):
                    stats['n_ignore_long'] += 1
                    continue

                stats["num_examples"] += 1
                # take max of sentence lengths to determine bucket that it goes in
                #buck_idx = 0 if len_s1 > len_s2 else 1

                id1 = np.searchsorted([x[0] for x in buckets], len_s1)
                id2 = np.searchsorted([x[1] for x in buckets], len_s2)
                bucket_id = max(id1,id2)

                s1_ids = vocab.ids_for_tokens(s1, update_vocab)
                s2_ids = vocab.ids_for_tokens(s2, update_vocab)

                # pad using the bucket length tuples for premise and hypothesis
                s1_f = pad_sentence(s1_ids, pad_length=buckets[bucket_id][0], pad_id=vocab.PAD_ID)
                s2_f = pad_sentence(s2_ids, pad_length=buckets[bucket_id][1], pad_id=vocab.PAD_ID)


                bucket_dict[bucket_id]["s1_batch"].append(s1_f)
                bucket_dict[bucket_id]["s2_batch"].append(s2_f)
                bucket_dict[bucket_id]["labels"].append(label)

                # flush batch
                if len(bucket_dict[bucket_id]["s1_batch"]) == batch_size:


                    sents = {"premise": np.asarray(bucket_dict[bucket_id]["s1_batch"])
                        .reshape(batch_size, buckets[bucket_id][0]),
                         "hypothesis": np.asarray(bucket_dict[bucket_id]["s2_batch"])
                             .reshape(batch_size, buckets[bucket_id][1])}
                    tar = np.asarray(bucket_dict[bucket_id]["labels"]).reshape(batch_size, 3)

                    output[bucket_id].append((sents,tar))
                    bucket_dict[bucket_id]["s1_batch"] = []
                    bucket_dict[bucket_id]["s2_batch"] = []
                    bucket_dict[bucket_id]["labels"] = []

                # break if the current largest bucket is greater than the max allowed
                if max_records and (max([len(x) for x in output]) == max_records):
                    break

        all_output.append(output)
        all_stats.append(stats)

    train_data, val_data, test_data = all_output
    return  train_data, val_data, test_data, all_stats



def bucket_shuffle(dict_data):
    # zip each data tuple with it's bucket id.
    # return as a randomly shuffled iterator.
    id_to_data =[]
    for x, data in dict_data.items():
        id_to_data += list(zip([x]*len(data), data))

    shuffle(id_to_data)

    return len(id_to_data), iter(id_to_data)



if __name__=="__main__":
    import sys
    import pickle

    # test for loader
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    vocab = Vocab("/Users/markneumann/Documents/Machine_Learning/act-rte-inference/snli_1.0/debug_vocab.txt", dataset_path, 30000)

    buckets = [(10,5),(20,10),(30,20),(40,30),(50,40),(82,62)]
    raw_data = load_data(args.dataset_path,"snli_1.0_train.jsonl","snli_1.0_dev.jsonl","snli_1.0_test.jsonl", vocab, False,
                            max_records=None, buckets=buckets, batch_size=32)

    train, dev, test, stats = raw_data

    for bucket in train:
        print(len(bucket))


    # all_out = load_data(dataset_path,"snli_1.0_train.jsonl","snli_1.0_dev.jsonl","snli_1.0_test.jsonl",
    #                         vocab, False,buckets=[(10,10),(15,20)],max_records=100, max_len=(60,60), batch_size=30)
    #
    # train, dev, test, stats = all_out
    #
    # train_dict = {x:v for x,v in enumerate(train)}
    #
    # buckets = [(10,10), (15,20), (60,60)]
    # len, it = bucket_shuffle(train_dict)
    #
    # for (id, data) in it:
    #
    #      assert data[0]["hypothesis"].shape == (30,buckets[id][1])
    #      assert data[0]["premise"].shape == (30,buckets[id][0])

