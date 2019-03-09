import os
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
from pathlib import Path
import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / 'fashion' / 'MODEL NAME')
TRAIN_DIR = str(psychic_learners_dir / 'data' / 'image' / 'v1_train_240x240' / 'fashion')
VAL_DIR = str(psychic_learners_dir / 'data' / 'image' / 'valid_240x240' / 'fashion')
IMAGE_SIZE = (240, 240)
N_CLASSES = 14
BATCH_SIZE = 64
model = keras.models.load_model(IMAGE_MODEL_PATH)
model = model.layers[:-1]  # TODO FIND CORRECT NUMBER OF LAYERS

train_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)
train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                          color_mode='rgb', batch_size=BATCH_SIZE, interpolation='bicubic')
valid = valid_datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                          color_mode='rgb', batch_size=BATCH_SIZE, interpolation='bicubic')
train_steps = len(train)
valid_steps = len(valid)
def get_train_image_features():
    train_img_features = model.predict_generator(train, steps=train_steps, callbacks=None,
                                       max_queue_size=10, workers=cpu_count(), 
                                       use_multiprocessing=True, verbose=1)
    np.savetxt('train_img_features.gz', train_img_features)

def get_valid_image_features():
    valid_img_features = model.predict_generator(valid, steps=valid_steps, callbacks=None,
                                       max_queue_size=10, workers=cpu_count(),
                                       use_multiprocessing=True, verbose=1)
    np.savetxt('valid_img_features.gz', valid_img_features)

def load_train_features():
    train_img_features = np.loadtxt('train_img_features.gz')
    print(train_img_features.shape)
    train_bert_features = np.loadtxt()
    print(train_bert_features.shape)
    return train_img_features, train_bert_features

def load_valid_features():
    valid_image_features = np.loadtxt('valid_img_features.gz')
    print(valid_image_features.shape)
    valid_bert_features = np.loadtxt()
    print(valid_bert_features.shape)
    return valid_image_features, valid_bert_features


class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64, dim=(32, 32, 32), n_channels=1,
                 n_classes=N_CLASSES, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_temp = [self.x[i] for i in indexes]

        # Generate data
        X, y = self.__data_generation(x_temp) # may not need this step

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(x_temp):
            # Store sample
            X[i, ] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.y[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

if __name__ == '__main__':
    get_train_image_features()
    get_valid_image_features()
