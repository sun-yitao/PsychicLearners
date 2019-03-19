from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

from random_eraser import get_random_eraser

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'beauty'
TRAIN_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'v1_train_240x240' / BIG_CATEGORY)
VALID_IMAGE_DIR = str(psychic_learners_dir / 'data' / 'image' / 'valid_240x240' / BIG_CATEGORY)
TEST_IMAGE_DIR = str(psychic_learners_dir / 'data' /  'test_240x240' / 'test_240x240')
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' /  f'{BIG_CATEGORY}_test_split.csv')
CHECKPOINT_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY)
IMAGE_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'converted_model.h5')
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'word_cnn' / '0.7805109797045588.ckpt-382000')
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
EPOCHS = 200  # only for calculation of decay
IMAGE_SIZE = (240, 240)  # height, width
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
STARTING_CLASS_FOR_CATEGORIES = {'beauty': 0, 'fashion': 17, 'mobile': 31}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
STARTING_CLASS = STARTING_CLASS_FOR_CATEGORIES[BIG_CATEGORY]
MODEL_NAME = f'{BIG_CATEGORY}_multi_inp_dnn_cnn'
LR_BASE = 0.01
LR_DECAY_FACTOR = 1
BATCH_SIZE = 64
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MultiInputDataGenerator(keras.utils.Sequence):
    """Takes in image generator and df, outputs image, title_seq, onehotencoded y
        image generator shuffle must be false"""
    def __init__(self, img_gen, titles_seq, batch_size=64, title_dim=16,
                 n_classes=N_CLASSES, shuffle=True):
        'Initialization'
        self.img_gen = img_gen
        self.titles_seq = titles_seq
        self.title_dim = title_dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.img_gen)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = np.arange(index*self.batch_size, (index+1)*self.batch_size)
        X_im = self.img_gen.__getitem__(index)[0]
        X_title_seq = self.titles_seq[batch_indexes]
        y = self.img_gen.__getitem__(index)[1]
        return [X_im, X_title_seq], [y, y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_gen)) #index 0 = first batch of img data generator
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_image_name(filepath):
    image_name = os.path.split(filepath)[-1]
    if not image_name.endswith('.jpg'):
        image_name += '.jpg'
    return image_name

def multi_input_model(image_model, vocab_size, k_reg=0):
    filter_sizes = (3, 8)
    k_regularizer = keras.regularizers.l2(k_reg)
    text_input = layers.Input(shape=(16,), name='text_input')
    embedding = layers.Embedding(input_dim=vocab_size,
                                 output_dim=128,
                                 input_length=vocab_size)(text_input)
    x = keras.layers.Flatten()(embedding)
    x = layers.Dense(128, activation=None, kernel_initializer='he_uniform',
                     kernel_regularizer=k_regularizer)(x)
    text_out = layers.PReLU()(x)
    text_output = layers.Dense(N_CLASSES, activation='softmax', name='text_output', kernel_regularizer=k_regularizer)(text_out)
    
    concat = keras.layers.concatenate([text_out, image_model.output])
    final_output = layers.Dense(N_CLASSES, activation='softmax',
                                name='final_output', kernel_regularizer=k_regularizer)(concat)
    mul_inp_model = keras.models.Model(inputs=[image_model.input, text_input],
                                       outputs=[final_output, text_output])
    return mul_inp_model

def train(train_gen, valid_gen, class_weights=None):
    image_model = keras.models.load_model(IMAGE_MODEL_PATH)
    image_model.layers.pop()  # remove fully connected layers
    image_model.layers.pop()  # remove fully connected layers
    image_model.layers.pop()  # remove pooling
    image_model = keras.models.Model(inputs=image_model.input, outputs=image_model.layers[-1].output)

    multi_inp_model = multi_input_model(image_model, vocab_size=vocab_size, k_reg=0)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    multi_inp_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                            loss_weights=[1., 0.2])
    # callbacks
    checkpoint_path = os.path.join(CHECKPOINT_PATH, '{}_checkpoints'.format(MODEL_NAME))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7,
                                                  verbose=1, mode='auto',  # min_delta=0.001,
                                                  cooldown=0, min_lr=0)

    multi_inp_model.fit_generator(multi_inp_train, steps_per_epoch=len(train_gen), epochs=1000,
                                  validation_data=valid_gen, validation_steps=len(valid_gen),
                                  callbacks=[ckpt, reduce_lr])  # class_weight=class_weights)

def extract_probs(valid_gen, test_gen, model_checkpoint):
    model = keras.models.load_model(model_checkpoint)
    valid_preds = model.predict_generator(valid_gen, steps=len(valid_gen), workers=1, use_multiprocessing=False, verbose=1)
    test_preds = model.predict_generator(test_gen, steps=len(test_gen), workers=1, use_multiprocessing=False, verbose=1)
    print(valid_preds.shape)
    print(test_preds.shape)
    os.makedirs(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'multi_inp'))
    np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'multi_inp' / 'valid.npy'), valid_preds)
    np.save(str(ROOT_PROBA_FOLDER / BIG_CATEGORY / 'multi_inp' / 'test.npy'), test_preds)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # input generators
    # preprocessing function executes before rescale
        
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_df['image_filename'] = train_df['image_path'].map(get_image_name)
    train_df['image_filename'] = train_df.apply(lambda df: os.path.join(str(df['Category']), df['image_filename']), axis=1)
    valid_df['image_filename'] = valid_df['image_path'].map(get_image_name)
    valid_df['image_filename'] = valid_df.apply(lambda df: os.path.join(str(df['Category']), df['image_filename']), axis=1)
    test_df['image_filename'] = test_df['image_path'].map(get_image_name)

    img_train_datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.1,
                                       height_shift_range=0.1, brightness_range=(0.9, 1.1),
                                       shear_range=0.0, zoom_range=0.1,
                                       channel_shift_range=0.2,
                                       fill_mode='reflect', horizontal_flip=True,
                                       vertical_flip=False, rescale=1/255,
                                       preprocessing_function=get_random_eraser(p=0.8, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                                                                                v_l=0, v_h=255, pixel_level=True))
    img_valid_datagen = ImageDataGenerator(rescale=1/255)
    img_test_datagen = ImageDataGenerator(rescale=1/255)
    train = img_train_datagen.flow_from_dataframe(train_df, directory=TRAIN_IMAGE_DIR, x_col='image_filename', y_col='Category',
                                                  target_size=IMAGE_SIZE, color_mode='rgb', classes=None, class_mode=None,
                                                  batch_size=BATCH_SIZE, shuffle=False, seed=101, interpolation='bicubic')

    valid = img_valid_datagen.flow_from_dataframe(valid_df, directory=VALID_IMAGE_DIR, x_col='image_filename', y_col='Category',
                                                    target_size=IMAGE_SIZE, color_mode='rgb', classes=None, class_mode=None,
                                                    batch_size=BATCH_SIZE, shuffle=False, seed=101, interpolation='bicubic')
    test = img_test_datagen.flow_from_dataframe(test_df, directory=TEST_IMAGE_DIR, x_col='image_filename', y_col=None,
                                                target_size=IMAGE_SIZE, color_mode='rgb', classes=None, class_mode=None,
                                                batch_size=BATCH_SIZE, shuffle=False, seed=101, interpolation='bicubic')
    #class_weights = compute_class_weight('balanced', np.arange(0, N_CLASSES), train.labels)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['title'].values)
    train_titles_seq = tokenizer.texts_to_sequences(train_df['title'].values)
    valid_titles_seq = tokenizer.texts_to_sequences(valid_df['title'].values)
    test_titles_seq = tokenizer.texts_to_sequences(test_df['title'].values)
    train_titles_seq = pad_sequences(train_titles_seq, padding='post', maxlen=16)
    valid_titles_seq = pad_sequences(valid_titles_seq, padding='post', maxlen=16)
    test_titles_seq = pad_sequences(test_titles_seq, padding='post', maxlen=16)
    vocab_size = len(tokenizer.word_index) + 1

    multi_inp_train = MultiInputDataGenerator(train, train_titles_seq, batch_size=64, title_dim=16,
                                              n_classes=N_CLASSES, shuffle=True)
    multi_inp_valid = MultiInputDataGenerator(valid, valid_titles_seq, batch_size=64, title_dim=16,
                                              n_classes=N_CLASSES, shuffle=False)
    multi_inp_test = MultiInputDataGenerator(test, test_titles_seq, batch_size=64, title_dim=16,
                                              n_classes=N_CLASSES, shuffle=False)

    train(multi_inp_train, multi_inp_valid)
    
