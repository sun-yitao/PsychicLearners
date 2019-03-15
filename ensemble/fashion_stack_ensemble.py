import os
from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf

"""Stacking Ensemble using probabilties predicted on validation, validating on public test set
    probs from ml-ensemble, fasttext, bert, combined-features classifier"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'fashion'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'MODEL NAME')
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities' / BIG_CATEGORY)
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES = 14
N_MODELS = 3
BATCH_SIZE = 64
image_model = keras.models.load_model(IMAGE_MODEL_PATH)
image_model = image_model.layers[:-1]  # TODO FIND CORRECT NUMBER OF LAYERS
print(image_model.summary())

def read_probabilties(proba_folder, subset='valid',
                      model_names=None):
    proba_folder = Path(proba_folder)
    all_probabilities = []
    for folder in proba_folder.iterdir():
        if not folder.is_dir():
            continue
        elif model_names and folder.name not in model_names:
            continue
        for npy in folder.glob(f'*{subset}.npy'):
            prob = np.load(str(npy))
            print(prob.shape)
            all_probabilities.append(prob)

    #TODO Concatenate probs along axis
    for prob in all_probabilities:
        pass
    return all_probabilities


MODEL_INPUT_SHAPE = (N_CLASSES * N_MODELS)
def ensemble_model(dense1=None, dense2=None, dropout=0.25, k_reg=0.0001):
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    if dense1:
        x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform')(input_tensor)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
    if dense2:
        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform')(input_tensor)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
    if dense1:
        predictions = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(x)
    else:
        predictions = layers.Dense(
            N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(input_tensor)
    model = keras.models.Model(inputs=input_tensor, outputs=predictions)
    return model


def train(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined'),
          model_name='1'):
    model = ensemble_model(dense1=1024, dense2=None,
                           dropout=0.25, k_reg=0.0001)
    decay = lr_base/(epochs * lr_decay_factor)
    sgd = keras.optimizers.SGD(
        lr=lr_base, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model_{}_checkpoints'.format(model_name))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                                  verbose=1, mode='auto',
                                                  cooldown=0, min_lr=0)
    log_dir = "logs_fashion/combined_model_{}".format(model_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    train_x = read_probabilties(proba_folder='', subset='valid') #TODO check shuffling
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    encoder = OneHotEncoder(sparse=False)
    train_y = encoder.fit_transform(train_y.reshape(-1, 1))
    test_x = read_probabilties(proba_folder='', subset='test')

    model.fit(x=train_x, y=train_y, batch_size=BATCH_SIZE, epochs=epochs, verbose=2, 
              callbacks=[ckpt, reduce_lr, tensorboard], validation_split=0.2,
              shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)
    
if __name__ == '__main__':
    
