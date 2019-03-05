from datetime import datetime
import os
import tensorflow as tf
import keras
from keras.layers import Dense, Input
from random_eraser import get_random_eraser
from se_inception_resnet_v2 import SEInceptionResNetV2
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import multi_gpu_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

psychic_learners_dir = os.path.split(os.getcwd())[0]
TRAIN_DIR = os.path.join(psychic_learners_dir, 'data',
                         'image', 'v1_train_nodups_240x240', 'mobile')
VAL_DIR = os.path.join(psychic_learners_dir, 'data', 'image', 'valid_240x240', 'mobile')
CHECKPOINT_PATH = os.path.join(psychic_learners_dir, 'data', 'keras_checkpoints', 'mobile')
EPOCHS = 200  # only for calculation of decay
IMAGE_SIZE = (240, 240)  # height, width
N_CLASSES = 27
MODEL_NO = 1
LR_BASE = 0.001
LR_DECAY_FACTOR = 1
BATCH_SIZE = 64

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

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    # input generators
    train_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,
                                       height_shift_range=0.1, brightness_range=(0.9, 1.1),
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
    base_model = SEInceptionResNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                     include_top=False,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     pooling='avg',
                                     classes=N_CLASSES)
    x = base_model.output
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    model = ModelMGPU(model, 2)
    decay = LR_BASE/(EPOCHS * LR_DECAY_FACTOR)
    sgd = keras.optimizers.SGD(lr=LR_BASE, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    checkpoint_path = os.path.join(
        CHECKPOINT_PATH, 'model_{}_checkpoints'.format(MODEL_NO))
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7,
                                                  verbose=1, mode='auto',# min_delta=0.001,
                                                  cooldown=0, min_lr=0)
    log_dir = "logs_mobile/model_{}_{}".format(
        MODEL_NO, datetime.utcnow().strftime("%d%m%Y_%H%M%S"))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tensorboard = keras.callbacks.TensorBoard(log_dir)

    model.fit_generator(train, steps_per_epoch=train.n/train.batch_size, epochs=1000,
                        validation_data=valid, validation_steps=valid.n/valid.batch_size,
                        callbacks=[ckpt, reduce_lr, tensorboard])
