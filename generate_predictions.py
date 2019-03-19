from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pathlib import Path
from fastText import train_supervised, train_unsupervised, load_model
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    psychic_learners_dir = Path.cwd()
    #beauty
    #train_data = os.path.join(psychic_learners_dir, 'data', 'beauty_train_split.txt')
    #beauty_model = train_supervised(
    #    input=train_data, epoch=6, lr=0.1, dim=100, 
    #    wordNgrams=4, verbose=2, ws=5, lrUpdateRate=100,
    #)
    beauty_model = load_model(str(psychic_learners_dir / 'data' / 'beauty_model.bin'))

    #fashion
    #train_data = os.path.join(psychic_learners_dir, 'data', 'fashion_train_split.txt')
    #fashion_model = train_supervised(
    #    input=train_data, epoch=8, lr=0.1, dim=120,
    #    wordNgrams=6, verbose=2, ws=8, lrUpdateRate=100,
    #)
    fashion_model = load_model(str(psychic_learners_dir / 'data' / 'fashion_model.bin'))
    
    #mobile
    #train_data = os.path.join(psychic_learners_dir, 'data', 'mobile_train_split.txt')
    #model = train_supervised(
    #    input=train_data, epoch=16, lr=0.1, dim=140,
    #    wordNgrams=4, verbose=2, ws=5, lrUpdateRate=100,
    #)
    mobile_model = load_model(str(psychic_learners_dir / 'data' / 'mobile_model.bin'))
    
    pred_df = pd.DataFrame(columns=['itemid', 'Category'])
    itemids = []
    predictions = []
    for big_category in ['beauty', 'fashion', 'mobile']:
        if big_category == 'beauty':
            model = beauty_model
        elif big_category == 'fashion':
            model = fashion_model
        elif big_category == 'mobile':
            model = mobile_model

        test_df = pd.read_csv(str(psychic_learners_dir / 'data' / f'{big_category}_test_split.csv'))
        for row in tqdm(test_df.itertuples()):
            itemid = int(row[1])
            itemids.append(itemid)
            title = row[2]
            pred = model.predict(title)
            pred = pred[0][0].replace('__label__', '')
            predictions.append(int(pred))
    pred_df = pd.DataFrame(data={'itemid': itemids, 'Category': predictions})
    pred_df.to_csv(str(psychic_learners_dir / 'data' / 'predictions_v1.csv'), index=False)


    

