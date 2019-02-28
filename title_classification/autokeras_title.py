import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import autokeras
from autokeras.text import text_preprocessor, text_supervised

data_directory = os.path.join('..', 'data')
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'valid_split.csv'))
train_x, train_y = train['title'], train['Category']
valid_x, valid_y = valid['title'], valid['Category']

clf = text_supervised.TextClassifier(verbose=True, path=os.getcwd())
clf.max_seq_length=7
clf.fit(x=train_x, y=train_y, time_limit=10 * 60 * 60)
print("Classification accuracy is : ", 100 * clf.evaluate(valid_x, valid_y), "%")
