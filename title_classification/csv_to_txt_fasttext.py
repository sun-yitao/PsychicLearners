import os
import csv
import pandas as pd

psychic_learners_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(psychic_learners_dir, 'data')
train = pd.read_csv(os.path.join(data_dir, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_dir, 'valid_split.csv'))
train['formatted_category'] = train['Category'].map(lambda x: '__label__' + str(x))
valid['formatted_category'] = valid['Category'].map(lambda x: '__label__' + str(x))

new_train = train[['formatted_category', 'title']]
new_valid = valid[['formatted_category', 'title']]

new_train.to_csv(os.path.join(data_dir, '_train_split.txt'),
                 header=None, index=None, sep=' ', mode='a')
new_valid.to_csv(os.path.join(data_dir, '_valid_split.txt'),
                 header=None, index=None, sep=' ', mode='a')

with open(os.path.join(data_dir, '_train_split.txt'), 'r') as infile, \
        open(os.path.join(data_dir, 'beauty_train_split.txt'), 'w') as outfile:
    data = infile.read()
    data = data.replace('"', '')
    outfile.write(data)

with open(os.path.join(data_dir, '_valid_split.txt'), 'r') as infile, \
        open(os.path.join(data_dir, 'beauty_valid_split.txt'), 'w') as outfile:
    data = infile.read()
    data = data.replace('"', '')
    outfile.write(data)
