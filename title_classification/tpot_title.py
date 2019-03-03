import os
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
import multiprocessing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
multiprocessing.set_start_method('forkserver')

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'valid_split.csv'))
train_x, train_y = train['title'].values, train['Category'].values
valid_x, valid_y = valid['title'].values, valid['Category'].values

tfidf_vect = TfidfVectorizer(analyzer='word', strip_accents='unicode',
                             stop_words='english', token_pattern=r'\w{1,}')
X_train = tfidf_vect.fit_transform(train['title'])
X_valid = tfidf_vect.fit_transform(valid['title'])
#tfidf_vect_ngram = TfidfVectorizer(analyzer='word', strip_accents='unicode',
#                                   stop_words='english', token_pattern=r'\w{1,}',
#                                   ngram_range=(1, 3))
#tfidf_vect_ngram.fit(train['title'])
y_train = to_categorical(train_y)
y_valid = to_categorical(valid_y)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20,
                                    offspring_size=None, mutation_rate=0.9,
                                    crossover_rate=0.1,
                                    scoring='accuracy', cv=5,
                                    subsample=1.0, n_jobs=-1,
                                    max_time_mins=10, max_eval_time_mins=5,
                                    random_state=42, config_dict='TPOT sparse',
                                    warm_start=False,
                                    memory='auto',
                                    use_dask=True,
                                    periodic_checkpoint_folder=None,
                                    early_stop=None,
                                    disable_update_check=False, 
                                    verbosity=3)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_valid, y_valid))
