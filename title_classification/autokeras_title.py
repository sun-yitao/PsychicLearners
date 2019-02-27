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
count_vect = CountVectorizer(analyzer='word', strip_accents='unicode',  # Stop words may not be needed as they seem to be already removed
                             stop_words='english', token_pattern=r'\w{1,}')
train_x = count_vect.fit_transform(train_x)
valid_x = count_vect.fit_transform(valid_x)

clf = text_supervised.TextClassifier(verbose=True, path=os.getcwd())
clf.fit(x=train_x, y=train_y, time_limit=10 * 60 * 60)
print("Classification accuracy is : ", 100 * clf.evaluate(valid_x, valid_y), "%")
