from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pathlib import Path
from ast import literal_eval
from fastText import train_supervised, train_unsupervised, load_model
import numpy as np
import pandas as pd



BIG_CATEGORY = 'mobile'
N_CLASSES = 27

if __name__ == "__main__":
    psychic_learners_dir = Path.cwd().parent
    valid_data = os.path.join(psychic_learners_dir, 'data', f'{BIG_CATEGORY}_valid_split.csv')
    test_data = os.path.join(psychic_learners_dir, 'data', f'{BIG_CATEGORY}_test_split.csv')
    ROOT_PROBA_FOLDER = os.path.join(psychic_learners_dir, 'data', 'probabilities', BIG_CATEGORY, 'extractions_fasttext')
    valid_data = pd.read_csv(valid_data)
    test_data = pd.read_csv(test_data)
    model = load_model(str(psychic_learners_dir / 'data' / 'fasttext_models' / f'{BIG_CATEGORY}_extractions_model.bin'))
    valid_preds = []
    test_preds = []
    for title in valid_data['extractions'].values:
        if title == '0':  # comment out if normal title
            valid_preds.append(np.zeros(N_CLASSES))
            continue
        title = ' '.join(literal_eval(title))   # comment out if normal title
        pred = model.predict(title, k=N_CLASSES)
        pred = sorted(zip(pred[0], pred[1]), key=lambda x: x[0])
        pred = [x[1] for x in pred]
        valid_preds.append(pred)
    
    for title in test_data['extractions'].values:
        if title == '0':  # comment out if normal title
            test_preds.append(np.zeros(N_CLASSES))
            continue
        title = ' '.join(literal_eval(title))  # comment out if normal title
        pred = model.predict(title, k=N_CLASSES)
        pred = sorted(zip(pred[0], pred[1]), key=lambda x: x[0])
        pred = [x[1] for x in pred]
        test_preds.append(pred)
    
    valid_preds = np.array(valid_preds)
    test_preds = np.array(test_preds)
    print(valid_preds.shape)
    print(test_preds.shape)
    os.makedirs(os.path.join(ROOT_PROBA_FOLDER), exist_ok=True)
    np.save(os.path.join(ROOT_PROBA_FOLDER, 'valid.npy'), valid_preds)
    np.save(os.path.join(ROOT_PROBA_FOLDER, 'test.npy'), test_preds)
    
    

