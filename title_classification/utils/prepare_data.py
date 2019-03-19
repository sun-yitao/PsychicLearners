import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    x = pd.Series(csv_file["title"])
    return x


def data_preprocessing(train, valid, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_valid = vocab_processor.transform(valid)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_valid_list = list(x_transform_valid)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_valid = np.array(x_valid_list)
    x_test = np.array(x_test_list)
    return x_train, x_valid, x_test, vocab, vocab_size


def data_preprocessing_v2(train, valid, test, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    valid_idx = tokenizer.texts_to_sequences(valid)
    test_idx = tokenizer.texts_to_sequences(test)
    valid_padded = pad_sequences(valid_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return valid_padded, test_padded, max_words + 2


def data_preprocessing_with_dict(train, valid, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    valid_idx = tokenizer.texts_to_sequences(valid)
    test_idx = tokenizer.texts_to_sequences(test)
    valid_padded = pad_sequences(valid_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return valid_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def fill_feed_dict(data_X, batch_size):
    """Generator to yield batches"""
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch
