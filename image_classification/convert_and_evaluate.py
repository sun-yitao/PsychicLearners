from datetime import datetime
import os
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

psychic_learners_dir = os.path.split(os.getcwd())[0]
TRAIN_DIR = os.path.join(psychic_learners_dir, 'data', 'image', 'v1_train_nodups_240x240', 'mobile')
VAL_DIR = os.path.join(psychic_learners_dir, 'data', 'image', 'valid_240x240', 'mobile')
CHECKPOINT_PATH = os.path.join(psychic_learners_dir, 'data', 'keras_checkpoints', 'mobile', 'mobile_weights.h5')
EPOCHS = 200 # only for calculation of decay
IMAGE_SIZE = (240, 240)  # height, width
N_CLASSES = 27
LR_DECAY_FACTOR = 1
BATCH_SIZE = 64
LR_BASE = 0.01
LR_DECAY_FACTOR = 1

"""We ran into problems with models trained on python 3.5 on GCP so this script helps to convert the models to python 3.6 and test for correctness"""

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    valid_datagen = ImageDataGenerator(rescale=1/255)
    valid = valid_datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                              color_mode='rgb', batch_size=BATCH_SIZE, interpolation='bicubic')
    # model
    
    input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = InceptionResNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                     include_top=False,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     pooling='avg',
                                     classes=N_CLASSES)
    x = base_model.output
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    #dummy = Dense(N_CLASSES, activation=None)(predictions)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    decay = LR_BASE/(EPOCHS * LR_DECAY_FACTOR)
    sgd = keras.optimizers.SGD(lr=LR_BASE, decay=1e-6, momentum=0.9, nesterov=True)
    #model = keras.models.load_model(CHECKPOINT_PATH)
    model.load_weights(CHECKPOINT_PATH)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.save('mobile_epoch9_70.h5')
    """
    model.evaluate_generator(valid, steps=len(valid), callbacks=None,
                   max_queue_size=10, workers=cpu_count(), use_multiprocessing=True, verbose=1)"""
