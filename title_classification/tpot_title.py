import os
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
multiprocessing.set_start_method('forkserver')

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'valid_split.csv'))
train_x, train_y = train['title'].values, train['Category'].values
valid_x, valid_y = valid['title'].values, valid['Category'].values

tfidf_vect = TfidfVectorizer(analyzer='word', strip_accents='unicode', max_features=10000,
                             stop_words='english', token_pattern=r'\w{1,}')
X_train = tfidf_vect.fit_transform(train_x).A
X_valid = tfidf_vect.fit_transform(valid_x).A
#tfidf_vect_ngram = TfidfVectorizer(analyzer='word', strip_accents='unicode',
#                                   stop_words='english', token_pattern=r'\w{1,}',
#                                   ngram_range=(1, 3))
#tfidf_vect_ngram.fit(train['title'])
y_train = train_y.reshape(-1, 1)
y_valid = valid_y.reshape(-1, 1)
y_train = OneHotEncoder(sparse=False).fit_transform(y_train)
y_valid = OneHotEncoder(sparse=False).fit_transform(y_train)
print(X_train.shape)
print(y_train.shape)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20,
                                    offspring_size=None, mutation_rate=0.9,
                                    crossover_rate=0.1,
                                    scoring='accuracy', cv=5,
                                    subsample=1.0, n_jobs=-1,
                                    max_time_mins=10, max_eval_time_mins=5,
                                    random_state=42, config_dict='TPOT light',
                                    warm_start=False,
                                    memory='auto',
                                    use_dask=True,
                                    periodic_checkpoint_folder=None,
                                    early_stop=None,
                                    disable_update_check=True, 
                                    verbosity=3)
pipeline_optimizer.fit(X_train, train_y)
print(pipeline_optimizer.score(X_valid, valid_y))
