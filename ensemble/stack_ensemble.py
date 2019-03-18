import os
from multiprocessing import cpu_count
from pathlib import Path

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import train_test_split

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # workaround for macOS mkl issue
"""Stacking Ensemble using probabilties predicted on validation, validating on public test set
    probs from ml-ensemble, fasttext, bert, combined-features classifier"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'mobile'
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
N_MODELS = 8
BATCH_SIZE = 64
model_names = ['char_cnn',
               'extractions_fasttext',
               'image_model',
               'title_fasttext',
               'word_cnn',
               'word_rnn',
               'rcnn',
               'bert_v1'
               ]

def read_probabilties(proba_folder, subset='valid',
                      model_names=model_names):
    proba_folder = Path(proba_folder)
    all_probabilities = []
    for folder in proba_folder.iterdir():
        if not folder.is_dir():
            continue
        elif model_names and folder.name not in model_names:
            print(folder.name, 'not included')
            continue
        for npy in folder.glob(f'*{subset}.npy'):
            prob = np.load(str(npy))
            print(prob.shape)
            prob = normalize(prob)
            all_probabilities.append(prob)

    all_probabilities = np.concatenate([prob for prob in all_probabilities], axis=1)
    print(all_probabilities.shape)
    return all_probabilities


MODEL_INPUT_SHAPE = (N_CLASSES * N_MODELS,)
def ensemble_model(dense1=None, dense2=None, dropout=0.25, k_reg=0):
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    if dense1:
        x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(input_tensor)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
        #x = layers.BatchNormalization()(x)
    if dense2:
        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(x)
        x = layers.PReLU()(x)
        #x = layers.BatchNormalization()(x)

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
    model = ensemble_model(dense1=256, dense2=256,
                           dropout=0.2, k_reg=0.00001)
    decay = lr_base/(epochs * lr_decay_factor)
    sgd = keras.optimizers.SGD(
        lr=lr_base, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join(
        checkpoint_dir, '{}_checkpoints'.format(model_name))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                                  verbose=1, mode='auto',
                                                  cooldown=0, min_lr=0)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
                                                           stratify=train_y,
                                                           test_size=0.25, random_state=42)
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_valid = encoder.fit_transform(y_valid.reshape(-1, 1))
    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=1000, verbose=2, 
              callbacks=[ckpt, reduce_lr], validation_data=(X_valid, y_valid),
              shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)

def predict(model_path, big_category, model_names=model_names):
    test_x = read_probabilties(proba_folder=os.path.join(
        ROOT_PROBA_FOLDER, big_category), subset='test')
    model = keras.models.load_model(model_path)
    preds = model.predict(test_x)
    print(preds.shape)
    return preds

def predict_all():
    beauty_preds = predict(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/beauty/combined/7+bert_checkpoints/model.28-0.80.h5', big_category='beauty')
    beauty_preds = np.argmax(beauty_preds, axis=1)
    beauty_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'beauty_test_split.csv'))
    beauty_preds = pd.DataFrame(data={'itemid':beauty_test['itemid'].values, 
                                      'Category': beauty_preds})
    
    fashion_preds = predict(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/fashion/combined/7+bert_checkpoints/model.15-0.68.h5', big_category='fashion')
    fashion_preds = np.argmax(fashion_preds, axis=1)
    fashion_preds = fashion_preds + 17
    fashion_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'fashion_test_split.csv'))
    fashion_preds = pd.DataFrame(data={'itemid': fashion_test['itemid'].values,
                                       'Category': fashion_preds})

    mobile_preds = predict(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/mobile/combined/7+bert_checkpoints/model.16-0.85.h5', big_category='mobile')
    mobile_preds = np.argmax(mobile_preds, axis=1)
    mobile_preds = mobile_preds + 31
    mobile_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'mobile_test_split.csv'))
    mobile_preds = pd.DataFrame(data={'itemid': mobile_test['itemid'].values,
                                      'Category': mobile_preds})

    all_preds = pd.concat([beauty_preds, fashion_preds, mobile_preds], ignore_index=True)
    all_preds.to_csv(str(psychic_learners_dir / 'data' / 'predictions' /
                         '7+bert.csv'), index=False)

def evaluate_total_accuracy(val_beauty_acc, val_fashion_acc, val_mobile_acc, kaggle_public_acc):
    val_split = 0.25
    total_examples = 57317*val_split + 43941*val_split + 32065*val_split + 172402*0.3
    num_correct = 57317*val_split*val_beauty_acc + \
                  43941*val_split*val_fashion_acc + \
                  32065*val_split*val_mobile_acc + \
                  172402*0.3*kaggle_public_acc
    return num_correct/total_examples


if __name__ == '__main__':
    """
    train(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' /
                             'keras_checkpoints' / BIG_CATEGORY / 'combined'),
          model_name='7+bert')"""

    predict_all()
    #print(evaluate_total_accuracy(0.79651, 0.67459, 0.8366, 0.76235)) #5+wordrnn+rcnn
    #print(evaluate_total_accuracy(0.79860, 0.67886, 0.84633, 0.76235)) #7+bert


"""
Logs
4+charcnn 0.7629462331932415
5+wordrnn 0.7628749750297908





"""
