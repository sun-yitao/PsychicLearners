import os
from pathlib import Path
import pickle
import re

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'mobile'
ROOT_PROBA_FOLDER = psychic_learners_dir / 'data' / 'probabilities'
MODEL_NAME = 'char_cnn'
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' /
                      BIG_CATEGORY / MODEL_NAME / '0.7740911354581673.ckpt-20000')
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
BATCH_SIZE = 128
WORD_MAX_LEN = 15
CHAR_MAX_LEN = 150

with open(str(psychic_learners_dir / 'ensemble' / "word_dict.pickle"), "rb") as f:
    word_dict = pickle.load(f)


def build_word_dataset(df, word_dict, document_max_len):
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["title"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d))
                 * [word_dict["<pad>"]], x))
    return x


def build_char_dataset(df, document_max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "

    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    x = list(map(lambda content: list(map(lambda d: char_dict.get(
        d, char_dict["<unk>"]), content.lower())), df["title"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d))
                 * [char_dict["<pad>"]], x))
    return x

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text


def batch_iter(inputs, batch_size):
    inputs = np.array(inputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(inputs))
        yield inputs[start_index:end_index]


def extract_and_save_probs(df, subset):
    #test_x = build_word_dataset(df, word_dict, WORD_MAX_LEN)
    test_x = build_char_dataset(df, CHAR_MAX_LEN) #for char_cnn
    graph = tf.Graph()
    all_probs = []
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(TEXT_MODEL_PATH))
            saver.restore(sess, TEXT_MODEL_PATH)
            for op in graph.get_operations():
                print(str(op.name))

            x = graph.get_operation_by_name("x").outputs[0]
            y = graph.get_operation_by_name("fc-3/dense/BiasAdd").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            batches = batch_iter(test_x, BATCH_SIZE)
            for batch_x in batches:
                feed_dict = {
                    x: batch_x,
                    is_training: False
                }

                probs = sess.run(y, feed_dict=feed_dict)
                print('Probabilities shape: ', probs.shape)
                for prob in probs:
                    all_probs.append(prob)

    all_probs = np.array(all_probs)
    print(f'All probs shape: {all_probs.shape}')
    os.makedirs(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / MODEL_NAME), exist_ok=True)
    np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / MODEL_NAME / f'{subset}.npy'), all_probs)

if __name__ == '__main__':
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)
    extract_and_save_probs(valid_df, subset='valid')
    extract_and_save_probs(test_df, subset='test')


"""OUTOUT LAYERS:

char cnn: fc-3/dense/BiasAdd
"""
