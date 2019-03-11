import os
from multiprocessing import cpu_count

from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import keras
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'fashion'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'MODEL NAME')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'train_240x240')
FEATURES_DIR = psychic_learners_dir / 'data'/ 'features' / BIG_CATEGORY
IMAGE_SIZE = (240, 240)
N_CLASSES = 14
BATCH_SIZE = 64
model = keras.models.load_model(IMAGE_MODEL_PATH)
model = model.layers[:-1]  # TODO FIND CORRECT NUMBER OF LAYERS
print(model.summary())

def save_image_features(features, itemids):
    for feature, itemid in zip(features, itemids):
        output_path = str(FEATURES_DIR / f'{itemid}_image_feature.npy')
        np.save(output_path, feature)


def extract_and_save_text_features(titles, itemid_array):
    #TODO
    pass


def get_features(csv, test=False):
    train = pd.read_csv(csv)
    steps = len(train) / BATCH_SIZE
    for batch in np.array_split(train, steps):
        np_image_array = np.empty(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        itemid_array = []
        titles = []
        for n, row in enumerate(batch.itertuples()):
            itemid = row[1]
            title = row[2]
            if test:
                image_path = row[4]
            else:
                #category = row[3]
                image_path = row[5]
            itemid.append(itemid)
            titles.append(title)
            im = Image.open(os.path.join(IMAGE_DIR, image_path))
            np_image_array[n] = img_to_array(im)

        image_features = model.predict(np_image_array, batch_size=len(batch))
        save_image_features(image_features, itemid_array)
        extract_and_save_text_features(titles, itemid_array)




class DataGenerator(keras.utils.Sequence):
    #TODO change dims
    # train_datagen = DataGenerator(x=train['itemid'], y=train['Category'], batch_size=BATCH_SIZE)
    """x: itemid y:category sparse"""
    def __init__(self, x, y, batch_size=64, dim=(32, 32, 32),
                 n_classes=N_CLASSES, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.y = y
        self.x = x
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_temp = [self.x[i] for i in batch_indexes]

        # Generate data
        X, y = self.__data_generation(x_temp)

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
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(x_temp):
            # Store sample
            image_feature = np.load(str(FEATURES_DIR / f'{ID}_image_feature.npy'))
            text_feature = np.load(str(FEATURES_DIR / f'{ID}_text_feature.npy'))
            combined = np.concatenate((image_feature, text_feature), axis=None)
            print(combined.shape)
            X[i, ] = combined
            # Store class
            y[i] = self.y[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def ensemble_model():
    #TODO returns model
    pass

if __name__ == '__main__':
    get_features(TRAIN_CSV)
    get_features(VALID_CSV)
    get_features(TEST_CSV, test=True)
    #get_valid_features()
