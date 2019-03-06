import os
from PIL import Image
from datetime import datetime
import talos as ta

import tensorflow as tf
import keras
from keras import callbacks
from keras.layers import *
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.nasnet import NASNetLarge
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import multi_gpu_model

from random_eraser import get_random_eraser

psychic_learners_dir = os.path.split(os.getcwd())[0]
TRAIN_DIR = os.path.join(psychic_learners_dir, 'data', 'image', 'v1_train_nodups_240x240', 'fashion')
VAL_DIR = os.path.join(psychic_learners_dir, 'data', 'image', 'valid_240x240', 'fashion')
CHECKPOINT_PATH = os.path.join(psychic_learners_dir, 'data', 'keras_checkpoints', 'fashion')
EPOCHS = 100
N_CLASSES = 14
MODEL_NO = 3
BATCH_SIZE = 64

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
class ModelMGPU(keras.models.Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
           return getattr(self._smodel, attrname)
        else:
           #return Model.__getattribute__(self, attrname)
           return super(ModelMGPU, self).__getattribute__(attrname)

p = {
    # your parameter boundaries come here
    'preprocess': ['rescale', 'preprocess'],
    #'image_size': [144, 196, 240],
    #'model': ['inception_resnet_v2', 'nasnet'], 
    #'learning_rate': [0.1, 0.01],
    #'decay_factor': [1, 2, 5, 10, 100],
    #'momentum': [0.9, 0.95, 0.99],
    'deep_layers': [2, 3, 4],
    'freeze_layers': [1039, 1004, 668]
}

def input_model(x_train, y_train, x_val, y_val, params):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # input generators
    #IMAGE_SIZE = (params['image_size'], params['image_size'])
    IMAGE_SIZE = (240, 240)
    if params['preprocess'] == 'preprocess':
        train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,
                                           height_shift_range=0.1, brightness_range=(0.85, 1.15),
                                           shear_range=0.0, zoom_range=0.2,
                                           channel_shift_range=0.2,
                                           fill_mode='reflect', horizontal_flip=True,
                                           vertical_flip=False, preprocessing_function=preprocess_input)
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,
                                           height_shift_range=0.1, brightness_range=(0.85, 1.15),
                                           shear_range=0.0, zoom_range=0.2,
                                           channel_shift_range=0.2,
                                           fill_mode='reflect', horizontal_flip=True,
                                           vertical_flip=False, rescale=1/255,
                                           preprocessing_function=get_random_eraser(p=0.8, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                                                                                    v_l=0, v_h=255, pixel_level=True))
        valid_datagen = ImageDataGenerator(rescale=1/255)

    train = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE,
                                                color_mode='rgb', batch_size=BATCH_SIZE, interpolation='bicubic')
    valid = valid_datagen.flow_from_directory(VAL_DIR, target_size=IMAGE_SIZE,
                                                color_mode='rgb', batch_size=BATCH_SIZE, interpolation='bicubic')

    # model
    input_tensor = keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = NASNetLarge(include_top=False,
                             weights='imagenet',
                             input_tensor=input_tensor,
                             input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                             pooling=None,
                             classes=N_CLASSES)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(64, kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = Dropout(0.2)(x)
    x = Dense(64, kernel_initializer='he_normal',
              kernel_regularizer=keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    if params['deep_layers'] >= 3:
        x = Dropout(0.2)(x)
        x = Dense(64, kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

    if params['deep_layers'] == 4:
        x = Dropout(0.2)(x)
        x = Dense(64, kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

    predictions = Dense(N_CLASSES, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    model = ModelMGPU(model, 2)
    for layer in model.layers[:params['freeze']]:
        layer.trainable = False

    #LR_BASE = 0.01
    #decay = LR_BASE/(EPOCHS)
    #sgd = keras.optimizers.SGD(lr=LR_BASE, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}_checkpoints'.format(MODEL_NO))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                            verbose=1, mode='auto', min_delta=0.001,
                                            cooldown=0, min_lr=0)
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
    log_dir = "logs/model_{}_{}".format(MODEL_NO, datetime.utcnow().strftime("%d%m%Y_%H%M%S"))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tensorboard = callbacks.TensorBoard(log_dir)

    out = model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=EPOCHS,
                              validation_data=valid, validation_steps=valid.n/valid.batch_size,
                              callbacks=[ckpt, reduce_lr, early_stopping, tensorboard])

    return out, model


if __name__ == '__main__':
    # workaround to feed data in batches instead of loading into memory as talos requires
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
    ta.Deploy(h, 'talos_nasnet_transfer_model_search')
    print(h.data.head())
    print(h.details)
