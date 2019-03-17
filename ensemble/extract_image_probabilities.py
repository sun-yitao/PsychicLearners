import os
from pathlib import Path
from multiprocessing import cpu_count

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf
"""Load image model and extracts probabilties"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'beauty'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' /
                       BIG_CATEGORY / 'converted_model.h5')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_test_split.csv')
VALID_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'valid_240x240' / BIG_CATEGORY)
TEST_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'test_240x240')
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
IMAGE_SIZE = (240, 240)
N_CLASSES = 17
BATCH_SIZE = 1

def get_image_name(filepath):
    image_name = os.path.split(filepath)[-1]
    if not image_name.endswith('.jpg'):
        image_name += '.jpg'
    return image_name

valid_df = pd.read_csv(VALID_CSV)
test_df = pd.read_csv(TEST_CSV)
valid_df['image_filename'] = valid_df['image_path'].map(get_image_name)
valid_df['image_filename'] = valid_df.apply(lambda df: os.path.join(str(df['Category']), df['image_filename']), axis=1)
test_df['image_filename'] = test_df['image_path'].map(get_image_name)
print(valid_df['image_filename'].head(5))
valid_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
valid = valid_datagen.flow_from_dataframe(valid_df, directory=VALID_IMAGE_DIR, x_col='image_filename', y_col=None,
                                          target_size=IMAGE_SIZE, color_mode='rgb', classes=None, class_mode=None,
                                          batch_size=BATCH_SIZE, shuffle=False, seed=101, interpolation='bicubic')
test = test_datagen.flow_from_dataframe(test_df, directory=TEST_IMAGE_DIR, x_col='image_filename', y_col=None,
                                        target_size=IMAGE_SIZE, color_mode='rgb', classes=None, class_mode=None,
                                        batch_size=BATCH_SIZE, shuffle=False, seed=101, interpolation='bicubic')

model = keras.models.load_model(IMAGE_MODEL_PATH)
valid_preds = model.predict_generator(valid, steps=len(valid), workers=cpu_count(), use_multiprocessing=True, verbose=1)
test_preds = model.predict_generator(test, steps=len(test), workers=cpu_count(), use_multiprocessing=True, verbose=1)
print(valid_preds.shape)
print(test_preds.shape)
os.makedirs(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'image_model'))
np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'image_model' / 'valid.npy'), valid_preds)
np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'image_model' / 'test.npy'), test_preds)

