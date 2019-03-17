import os
from multiprocessing import cpu_count
from pathlib import Path
import pickle

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf

from data_utils import build_word_dataset, batch_iter
"""Combines text and image features to form one classifier"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'mobile'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'inceptres_imagent_weights' / 'mobile_epoch9_70.h5')
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' /
                      BIG_CATEGORY / 'word_cnn' / '0.8223667828685259.ckpt-686000')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'train_240x240')
FEATURES_DIR = psychic_learners_dir / 'data'/ 'features' / BIG_CATEGORY
IMAGE_SIZE = (240, 240)
N_CLASSES = 27
BATCH_SIZE = 64

#image_model = keras.models.load_model(IMAGE_MODEL_PATH)
#image_model = image_model.layers[:-1]  # TODO FIND CORRECT NUMBER OF LAYERS
#image_model = keras.models.Model(inputs=image_model.layers[0], outputs=image_model.output)
#print(image_model.summary())

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(TEXT_MODEL_PATH))
        saver.restore(sess, TEXT_MODEL_PATH)
        for op in graph.get_operations():
            print(str(op.name))

with open("word_dict.pickle", "rb") as f:
    word_dict = pickle.load(f)

def save_image_features(features, itemids):
    for feature, itemid in zip(features, itemids):
        output_path = str(FEATURES_DIR / f'{itemid}_image_feature.npy')
        np.save(output_path, feature)


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
    x = list(
        map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d))
                 * [word_dict["<pad>"]], x))

    y = list(map(lambda d: d-17, list(df["Category"])))

    return x, y

def extract_and_save_text_features(titles, itemid_array):
    test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN)
    checkpoint_file = TEXT_MODEL_PATH
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            x = graph.get_operation_by_name("x").outputs[0]
            y = graph.get_operation_by_name("y").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0
            for batch_x, batch_y in batches:
                feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    is_training: False
                }

                accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
                sum_accuracy += accuracy_out
                cnt += 1

            print("Test Accuracy : {0}".format(sum_accuracy / cnt))

    output_path = str(FEATURES_DIR / f'{itemid}_text_feature.npy')
    np.save(output_path, feature)
    pass


def get_features(csv, test=False):
    """Extracts and saves text and image features"""
    df = pd.read_csv(csv)
    steps = len(df) / BATCH_SIZE
    for batch in np.array_split(df, steps):
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
    get_features(TRAIN_CSV)
    get_features(VALID_CSV)
    get_features(TEST_CSV, test=True)
    #get_valid_features()
