import os
from pathlib import Path
import keras
from magpie import Magpie
import tensorflow as tf
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
keras.backend.set_session(session)

data_dir = Path.cwd().parent / 'data'
labels = [str(x) for x in range(17,31)]
ckpt_dir = data_dir / 'magpie' / 'checkpoints' / 'fashion' / 'v1'
os.makedirs(str(ckpt_dir / 'scaler.pkl'), exist_ok=True)

ckpt = keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                       monitor='val_acc', verbose=1, save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                              verbose=1, mode='auto',  # min_delta=0.001,
                                              cooldown=0, min_lr=0)

magpie = Magpie(word2vec_model=str(Path.cwd().parent/ 'title_classification' / 'word2vec.bin'))
magpie.fit_scaler(str(data_dir / 'magpie' / 'fashion' / 'train'))
magpie.save_scaler(str(ckpt_dir), overwrite=True)
magpie.batch_train(str(data_dir / 'magpie' / 'fashion' / 'train'), labels, 
                   test_dir=str(data_dir / 'magpie' / 'fashion' / 'valid'),
                   batch_size=256, epochs=30, callbacks=[ckpt, reduce_lr], verbose=2)


#magpie.save_scaler(str(ckpt_dir / 'scaler.pkl'), overwrite=True)
magpie.save_model(str(ckpt_dir / 'magpie.h5'))
magpie.save_word2vec_model(str(ckpt_dir / 'word2vec.pkl'))
