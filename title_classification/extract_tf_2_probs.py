import os
from pathlib import Path
import pickle
import re

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.prepare_data import data_preprocessing_with_dict, data_preprocessing_v2, load_data
#from utils.model_helper import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'beauty'
ROOT_PROBA_FOLDER = psychic_learners_dir / 'data' / 'probabilities'
MODEL_NAME = 'adv_abblstm'
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' /
                      BIG_CATEGORY / MODEL_NAME / '0.6236409034791471.ckpt-202000')
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
                f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
                f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
               f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
BATCH_SIZE = 128
WORD_MAX_LEN = 16
CHAR_MAX_LEN = 150




def batch_iter(inputs, batch_size):
    inputs = np.array(inputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(inputs))
        yield inputs[start_index:end_index]

def extract_and_save_probs(input, subset):
    graph = tf.Graph()
    all_probs = []
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(TEXT_MODEL_PATH))
            saver.restore(sess, TEXT_MODEL_PATH)
            for op in graph.get_operations()[:-2000]:
                print(str(op.name))

            x = graph.get_operation_by_name("Placeholder").outputs[0]
            y = graph.get_operation_by_name("output/dense/BiasAdd").outputs[0]
            keep_prob = graph.get_operation_by_name("Placeholder_2").outputs[0]

            batches = batch_iter(test_x, BATCH_SIZE)
            for batch_x in batches:
                feed_dict = {
                    x: batch_x,
                    keep_prob: 1.0
                }

                probs = sess.run(y, feed_dict=feed_dict)
                #print('Probabilities shape: ', probs.shape)
                for prob in probs:
                    all_probs.append(prob)

    all_probs = np.array(all_probs)
    print(f'All probs shape: {all_probs.shape}')
    os.makedirs(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / MODEL_NAME), exist_ok=True)
    np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / MODEL_NAME / f'{subset}.npy'), all_probs)
    test = all_probs[:10]
    test = np.argmax(test, axis=1)
    print(test)


if __name__ == '__main__':
    x_train = load_data(TRAIN_CSV)
    x_valid = load_data(VALID_CSV)
    x_test = load_data(TEST_CSV)

    if MODEL_NAME == 'adv_abblstm':
        x_valid, x_test, vocab_freq, word2idx, vocab_size = \
            data_preprocessing_with_dict(x_train, x_valid, x_test, max_len=WORD_MAX_LEN)
    else:
        x_valid, x_test, vocab_size = data_preprocessing_v2(x_train, x_valid, x_test, max_len=16)
        
    extract_and_save_probs(x_valid, subset='valid')
    extract_and_save_probs(x_test, subset='test')


"""OUTOUT LAYERS:

char cnn: fc-3/dense/BiasAdd
word rnn: output/dense/BiasAdd
"""
