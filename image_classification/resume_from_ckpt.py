from datetime import datetime
import os
import tensorflow as tf
import keras
from keras.layers import Dense, Input
#from keras.applications.xception import Xception, preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.resnext import ResNeXt101
from random_eraser import get_random_eraser
from se_resnext import SEResNextImageNet
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

psychic_learners_dir = os.path.split(os.getcwd())[0]
TRAIN_DIR = os.path.join(psychic_learners_dir, 'data',
                         'image', 'v1_train_240x240', 'fashion')
VAL_DIR = os.path.join(psychic_learners_dir, 'data',
                       'image', 'valid_240x240', 'fashion')
CHECKPOINT_PATH = os.path.join(
    psychic_learners_dir, 'data', 'keras_checkpoints', 'fashion')
SAVED_MODEL_PATH = '/home/paperspace/PsychicLearners/data/keras_checkpoints/fashion/model_3_checkpoints/model.42-0.53.h5'
EPOCHS = 200  # only for calculation of decay
IMAGE_SIZE = (240, 240)  # height, width
N_CLASSES = 14
MODEL_NO = 4
LR_BASE = 0.01
LR_DECAY_FACTOR = 1
BATCH_SIZE = 64

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # input generators
    train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,
                                       height_shift_range=0.1, brightness_range=(0.85, 1.15),
                                       shear_range=0.1, zoom_range=0.2,
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
    
    model = keras.models.load_model(SAVED_MODEL_PATH)
    decay = LR_BASE/(EPOCHS * LR_DECAY_FACTOR)
    sgd = keras.optimizers.SGD(lr=LR_BASE, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join(
        CHECKPOINT_PATH, 'model_{}_checkpoints'.format(MODEL_NO))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10,
                                                  verbose=1, mode='auto',  # min_delta=0.001,
                                                  cooldown=0, min_lr=0)
    #early_stopping = keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
    log_dir = "logs_fashion/model_{}_{}".format(
        MODEL_NO, datetime.utcnow().strftime("%d%m%Y_%H%M%S"))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tensorboard = keras.callbacks.TensorBoard(log_dir)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=1000,
                        validation_data=valid, validation_steps=valid.n/valid.batch_size,
                        callbacks=[ckpt, reduce_lr, tensorboard])
