import os
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import re

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
"""Combines text and image features to form one classifier"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'mobile'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'mobile_epoch9_70.h5')
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'word_cnn' / '0.8223667828685259.ckpt-686000')
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_test_split.csv')
TRAIN_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'v1_train_240x240' / BIG_CATEGORY)
VAL_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'valid_240x240' / BIG_CATEGORY)
TEST_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'test_240x240')
FEATURES_DIR = psychic_learners_dir / 'data'/ 'features' / BIG_CATEGORY
IMAGE_SIZE = (240, 240)
N_CLASSES = 27
BATCH_SIZE = 64
WORD_MAX_LEN = 15
os.makedirs(FEATURES_DIR, exist_ok=True)

image_model = keras.models.load_model(IMAGE_MODEL_PATH)
image_model.layers.pop()
image_model = keras.models.Model(inputs=image_model.input, outputs=image_model.layers[-1].output)
print(image_model.summary())

with open("word_dict.pickle", "rb") as f:
    word_dict = pickle.load(f)

def build_word_dataset(titles, word_dict, document_max_len):
    df = pd.DataFrame(data={'title': titles})
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["title"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
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

def extract_and_save_text_features(titles, itemid_array):
    """titles: array of titles"""
    test_x = build_word_dataset(titles, word_dict, WORD_MAX_LEN)
    graph = tf.Graph()
    all_text_features = []
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(TEXT_MODEL_PATH))
            saver.restore(sess, TEXT_MODEL_PATH)

            x = graph.get_operation_by_name("x").outputs[0]
            y = graph.get_operation_by_name("Reshape").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            batches = batch_iter(test_x, BATCH_SIZE)
            for batch_x in batches:
                feed_dict = {
                    x: batch_x,
                    is_training: False
                }

                text_features = sess.run(y, feed_dict=feed_dict)
                print('Text features shape: ', text_features.shape)
                for text_feature in text_features:
                    all_text_features.append(text_feature)
    for text_feature, itemid in zip(all_text_features, itemid_array):
        output_path = str(FEATURES_DIR / f'{itemid}_text_feature.npy')
        np.save(output_path, text_feature)


def save_image_features(features, itemids):
    for feature, itemid in zip(features, itemids):
        output_path = str(FEATURES_DIR / f'{itemid}_image_feature.npy')
        np.save(output_path, feature)


def get_features(csv, subset):
    """Extracts and saves text and image features"""
    df = pd.read_csv(csv)
    steps = (len(df) / BATCH_SIZE) + 1
    for batch in tqdm(np.array_split(df, steps)):
        np_image_array = np.empty((BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        itemid_array = []
        titles = []
        for n, row in enumerate(batch.itertuples()):
            itemid = row[1]
            title = row[2]
            if set == 'test':
                image_path = row[4]
            else:
                category = row[3]
                image_path = row[5]
            itemid_array.append(itemid)
            titles.append(title)
            image_path = os.path.split(image_path)[-1]
            if subset == 'test':
                im = Image.open(os.path.join(TEST_IMAGE_DIR, image_path))
            elif subset == 'valid':
                im = Image.open(os.path.join(VAL_IMAGE_DIR, str(category), image_path))
            elif subset == 'train':
                im = Image.open(os.path.join(TRAIN_IMAGE_DIR, str(category), image_path))
            np_image_array[n] = img_to_array(im)
        print(np_image_array.shape)
        image_features = image_model.predict(np_image_array, batch_size=len(batch))
        save_image_features(image_features, itemid_array)
        extract_and_save_text_features(titles, itemid_array)



MODEL_INPUT_SHAPE = (2048)
class DataGenerator(keras.utils.Sequence):
    #TODO change dims
    # Usage: train_datagen = DataGenerator(x=train['itemid'], y=train['Category'], batch_size=BATCH_SIZE)
    """x: concat features y:onehotencoded"""
    def __init__(self, x, y, batch_size=64, dim=MODEL_INPUT_SHAPE,
                 n_classes=N_CLASSES, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.encoder = OneHotEncoder(sparse=False)
        self.y = self.encoder.fit_transform(y.values.reshape(-1, 1))
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
        X = self.__data_generation(x_temp)
        y = [self.y[i] for i in batch_indexes]
        y = np.array(y)
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
        # Generate data
        for i, ID in enumerate(x_temp):
            # Store sample
            image_feature = np.load(str(FEATURES_DIR / f'{ID}_image_feature.npy'))
            text_feature = np.load(str(FEATURES_DIR / f'{ID}_text_feature.npy'))
            combined = np.concatenate((image_feature, text_feature), axis=None)
            print(combined.shape) #TODO check shape
            X[i, ] = combined
        return X

def combined_features_model(dense1=1024, dense2=None, dropout=0.25, k_reg=0.0001):
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform')(input_tensor)
    x = layers.PReLU()(x)
    x = layers.Dropout(dropout)(x)
    if dense2:
        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform')(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
    predictions = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(x)
    model = keras.models.Model(inputs=input_tensor, outputs=predictions)
    return model

def train_combined_model(lr_base=0.01, epochs=50, lr_decay_factor=1, 
                         checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined'),
                         model_name='1'):
    combined_model = combined_features_model(dense1=1024, dense2=None, dropout=0.25, k_reg=0.0001)
    decay = lr_base/(epochs * lr_decay_factor)
    sgd = keras.optimizers.SGD(lr=lr_base, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join(checkpoint_dir, 'model_{}_checkpoints'.format(model_name))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                                  verbose=1, mode='auto',
                                                  cooldown=0, min_lr=0)
    log_dir = "logs_fashion/combined_model_{}".format(model_name)
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir)


    combined_model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train = pd.read_csv(TRAIN_CSV)
    valid = pd.read_csv(VALID_CSV)
    train_datagen = DataGenerator(x=train['itemid'], y=train['Category'], batch_size=BATCH_SIZE)
    valid_datagen = DataGenerator(x=valid['itemid'], y=valid['Category'], batch_size=BATCH_SIZE)
    #class_weights = compute_class_weight('balanced', np.arange(0, N_CLASSES), train.classes)
    combined_model.fit_generator(train_datagen, steps_per_epoch=len(train_datagen), epochs=1000,
                                 validation_data=valid_datagen, validation_steps=len(valid_datagen),
                                 callbacks=[ckpt, reduce_lr, tensorboard])#class_weight=class_weights)

if __name__ == '__main__':
    get_features(TRAIN_CSV, subset='train')
    get_features(VALID_CSV, subset='valid')
    get_features(TEST_CSV, subset='test')
    #get_valid_features()
