import os
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np

BIG_CATEGORY = "beauty"
STARTING_CLASS = 0
TRAIN_PATH = f"{BIG_CATEGORY}_train_split.csv"
VALID_PATH = f"{BIG_CATEGORY}_valid_split.csv"
TEST_PATH = f"{BIG_CATEGORY}_test_split.csv"
titles_path = "titles.txt"

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        with open(titles_path, 'r') as f:
            contents = f.read().splitlines()

        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        df = pd.read_csv(TRAIN_PATH)
    elif step == "valid":
        df = pd.read_csv(VALID_PATH)
    else:
        df = pd.read_csv(TEST_PATH)

    # Shuffle dataframe
    df = df.sample(frac=1)
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["title"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    y = list(map(lambda d: d-17, list(df["Category"])))

    return x, y


def build_char_dataset(step, model, document_max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
    if step == "train":
        df = pd.read_csv(TRAIN_PATH)
    elif step == "valid":
        df = pd.read_csv(VALID_PATH)
    else:
        df = pd.read_csv(TEST_PATH)

    # Shuffle dataframe
    df = df.sample(frac=1)

    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), df["content"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    y = list(map(lambda d: d - 17, list(df["Category"])))

    return x, y, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
