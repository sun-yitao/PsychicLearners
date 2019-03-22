from pathlib import Path
import os
import numpy as np
import pandas as pd
from elmoformanylangs import Embedder
from nltk import word_tokenize

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'fashion'
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')

e = Embedder('/Users/sunyitao/Downloads/144/') # Path to pretrained elmo model
train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)
test_df = pd.read_csv(TEST_CSV)
train_df['tokenized'] = train_df['title'].apply(word_tokenize)
train_tokenized = train_df['tokenized'].values

valid_df['tokenized'] = valid_df['title'].apply(word_tokenize)
valid_tokenized = valid_df['tokenized'].values

test_df['tokenized'] = test_df['title'].apply(word_tokenize)
test_tokenized = test_df['tokenized'].values
all_tokenized = np.append(valid_tokenized, test_tokenized, axis=0)
print(all_tokenized.shape)
os.makedirs(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'elmo'), exist_ok=True)
out = e.sents2elmo(all_tokenized)
out = np.array(out)
np.save(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'elmo' / 'valid.npy'), out[:len(valid_tokenized)])
np.save(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY/  'elmo' / 'test.npy'), out[len(valid_tokenized):])
