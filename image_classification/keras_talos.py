import os
from PIL import Image
from datetime import datetime
import talos as ta

import tensorflow as tf
import keras
from keras import callbacks
from keras.layers import *
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnext import ResNeXt50, ResNeXt101
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

TRAIN_DIR = os.path.join('..', 'data', 'image', 'v1_train_240x240')
VAL_DIR = os.path.join('..', 'data', 'image', 'valid_240x240')
CHECKPOINT_PATH = os.path.join('..', 'data', 'keras_checkpoints')
EPOCHS = 100
N_CLASSES = 58
MODEL_NO = 2

p = {
    # your parameter boundaries come here
    'image_size': [100, 144, 196, 240],
    'model': ['xception', 'inception_resnet_v2', 'nasnet', 'resnext101'], 
    'learning_rate': [0.1, 0.5, 1.0],
    #'decay_factor': [1, 2, 5, 10, 100],
    #'momentum': [0.9, 0.95, 0.99],
}

def input_model(x_train, y_train, x_val, y_val, params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # input generators
    if params['model'] == 'NASNetLarge':
        batch_size = 64
    else:
        batch_size = 128
    IMAGE_SIZE = (params['image_size'], params['image_size'])
    train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.2,
                                       height_shift_range=0.2, brightness_range=(0.85, 1.15),
                                       shear_range=0.0, zoom_range=0.2,
                                       channel_shift_range=0.2,
                                       fill_mode='nearest', horizontal_flip=True,
                                       vertical_flip=False, preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                              color_mode='rgb', batch_size=batch_size, interpolation='bicubic')
    valid = valid_datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                              color_mode='rgb', batch_size=batch_size, interpolation='bicubic')

    # model
    input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    if params['model'] == 'xception':
        base_model = Xception(include_top=False,
                            weights=None,
                            input_tensor=input_tensor,
                            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                            pooling='avg',
                            classes=N_CLASSES)
    elif params['model'] == 'inception_resnet_v2':
        base_model = InceptionResNetV2(include_top=False,
                                       weights=None,
                                       input_tensor=input_tensor,
                                       input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                      pooling='avg',
                                       classes=N_CLASSES)
    elif params['model'] == 'nasnet':
        base_model = NASNetLarge(include_top=False,
                                 weights=None,
                                 input_tensor=input_tensor,
                                 input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                 pooling='avg',
                                 classes=N_CLASSES)
    elif params['model'] == 'resnext101':
        base_model = ResNeXt101(include_top=False,
                                 weights=None,
                                 input_tensor=input_tensor,
                                 input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                 pooling='avg',
                                 classes=N_CLASSES)
    else:
        base_model = ResNeXt50(include_top=False,
                                weights=None,
                                input_tensor=input_tensor,
                                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                pooling='avg',
                                classes=N_CLASSES)

    x = base_model.output
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    LR_BASE = params['learning_rate']
    decay = LR_BASE/(EPOCHS)
    sgd = keras.optimizers.SGD(lr=LR_BASE, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}_checkpoints'.format(MODEL_NO))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7,
                                            verbose=1, mode='auto', min_delta=0.001,
                                            cooldown=0, min_lr=0)
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=15)
    log_dir = "logs/model_{}_{}_{}".format(MODEL_NO, params['model'], datetime.utcnow().strftime("%d%m%Y_%H%M%S"))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tensorboard = callbacks.TensorBoard(log_dir)

    out = model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=EPOCHS,
                              validation_data=valid, validation_steps=valid.n/valid.batch_size,
                              callbacks=[ckpt, reduce_lr, early_stopping, tensorboard])

    return out, model


if __name__ == '__main__':
    x, y, x_val, y_val = [np.array([1, 2]) for i in range(4)]
    h = ta.Scan(x, y,
                params=p,
                model=input_model,
                #grid_downsample=.1,
                #reduction_method='correlation',
                x_val=x_val,
                y_val=y_val,
                val_split=0,
                shuffle=False,
                last_epoch_value=True,
                print_params=True)

    # accessing the results data frame
    print(h.data.head())
    # get the highest result ('val_acc' by default)
    print('Best accuracy: ', r.high())
    ta.Deploy(h, 'talos_prelim_model_search')
