import os
from pathlib import Path
from magpie import Magpie

data_dir = Path.cwd().parent / 'data'
labels = [str(x) for x in range(17,31)]
magpie = Magpie(word2vec_model=str(Path.cwd().parent/ 'title_classification' / 'word2vec.bin'))
magpie.fit_scaler(str(data_dir / 'magpie' / 'fashion' / 'train'))
magpie.train(str(data_dir / 'magpie' / 'fashion' / 'train'), labels, 
             test_dir=str(data_dir / 'magpie'/ 'fashion' / 'valid') , 
             batch_size=32, epochs=30, callbacks=None, verbose=2)

ckpt_dir = data_dir / 'magpie' / 'checkpoints' / 'fashion' / 'v1'
magpie.save_scaler(str(ckpt_dir), overwrite=True)
magpie.save_model(str(ckpt_dir / 'magpie.h5'))
magpie.save_word2vec_model(str(ckpt_dir))
