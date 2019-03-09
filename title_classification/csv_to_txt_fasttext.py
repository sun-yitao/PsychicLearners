import os
import csv
import pandas as pd

psychic_learners_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(psychic_learners_dir, 'data')
big_category = 'fashion'
train = pd.read_csv(os.path.join(data_dir, big_category + '_train_split.csv'))
valid = pd.read_csv(os.path.join(data_dir, big_category + '_valid_split.csv'))
#train['formatted_category'] = train['Category'].map(lambda x: '__label__' + str(x))
#valid['formatted_category'] = valid['Category'].map(lambda x: '__label__' + str(x))

#new_train = train[['formatted_category', 'translated_title']]
#new_valid = valid[['formatted_category', 'translated_title']]
new_train = train['translated_title']
new_valid = valid['translated_title']

new_train.to_csv(os.path.join(data_dir, '_train_split.txt'),
                 header=None, index=None, sep=' ', mode='a')
new_valid.to_csv(os.path.join(data_dir, '_valid_split.txt'),
                 header=None, index=None, sep=' ', mode='a')

with open(os.path.join(data_dir, '_train_split.txt'), 'r') as infile, \
        open(os.path.join(data_dir, big_category + '_train_sentences.txt'), 'w') as outfile:
    data = infile.read()
    data = data.replace('"', '')
    outfile.write(data)

with open(os.path.join(data_dir, '_valid_split.txt'), 'r') as infile, \
        open(os.path.join(data_dir, big_category + '_valid_sentences.txt'), 'w') as outfile:
    data = infile.read()
    data = data.replace('"', '')
    outfile.write(data)

os.remove(os.path.join(data_dir, '_train_split.txt'))
os.remove(os.path.join(data_dir, '_valid_split.txt'))
