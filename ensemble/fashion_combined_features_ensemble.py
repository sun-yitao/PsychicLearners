import os
from multiprocessing import cpu_count

from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf
"""Combines text and image features to form one classifier"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'fashion'
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'MODEL NAME' / 'model.h5')
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'MODEL NAME' / 'model.h5')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'train_240x240')
FEATURES_DIR = psychic_learners_dir / 'data'/ 'features' / BIG_CATEGORY
IMAGE_SIZE = (240, 240)
N_CLASSES = 14
BATCH_SIZE = 64
image_model = keras.models.load_model(IMAGE_MODEL_PATH)
image_model = image_model.layers[:-1]  # TODO FIND CORRECT NUMBER OF LAYERS
print(image_model.summary())
text_model = keras.models.load_model(TEXT_MODEL_PATH)
text_model = text_model.layers[:-1]

def save_image_features(features, itemids):
    for feature, itemid in zip(features, itemids):
        output_path = str(FEATURES_DIR / f'{itemid}_image_feature.npy')
        np.save(output_path, feature)


def extract_and_save_text_features(titles, itemid_array):
    #TODO do relevant preprocessing of title array and extract features
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
    """x: itemid y:category sparse"""
    def __init__(self, x, y, batch_size=64, dim=MODEL_INPUT_SHAPE,
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
            print(combined.shape) #TODO check shape
            X[i, ] = combined
            # Store class
            y[i] = self.y[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def combined_features_model(dense1=1024, dense2=None, dropout=0.25, k_reg=0.0001):
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform')(input_tensor)
    x = layers.PReLU()(x)
    x = layers.Dropout(dropout)(x)
    if dense2:
        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform')(input_tensor)
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
    combined_model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=1000,
                        validation_data=valid, validation_steps=valid.n/valid.batch_size,
                        callbacks=[ckpt, reduce_lr, tensorboard], class_weight=class_weights)

if __name__ == '__main__':
    get_features(TRAIN_CSV)
    get_features(VALID_CSV)
    get_features(TEST_CSV, test=True)
    #get_valid_features()
