'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification

 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 * Last Update: May 3rd, 2018
 * This file is part of  RMDL project, University of Virginia.
 * Free to use, change, share and distribute source code of RMDL
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import os
from RMDL import text_feature_extraction as txt
from sklearn.model_selection import train_test_split
from RMDL.Download import Download_WOS as WOS
import numpy as np
from RMDL import RMDL_Text as RMDL
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_data(file_name, sample_ratio=1, n_class=15, one_hot=True, starting_class=0):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["title"])
    y = pd.Series(shuffle_csv["Category"])
    y = y.astype(int) - starting_class
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y

def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]

if __name__ == "__main__":
    #dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')

    data_dir = r"D:\yitao\tcmtf"
    BIG_CATEGORY = 'beauty'
    N_CLASS = 17
    X_train, y_train = load_data(os.path.join(
        data_dir, f'{BIG_CATEGORY}_train_split.csv'), one_hot=False,n_class=N_CLASS, starting_class=0)
    X_test, y_test = load_data(os.path.join(
        data_dir, f'{BIG_CATEGORY}_valid_split.csv'), one_hot=False, n_class=N_CLASS, starting_class=0)



    batch_size = 100
    sparse_categorical = 0
    n_epochs = [500, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test,
                             batch_size=batch_size,
                             sparse_categorical=True,
                             random_deep=Random_Deep,
                             epochs=n_epochs)
