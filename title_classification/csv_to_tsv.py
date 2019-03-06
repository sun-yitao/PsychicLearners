import os
import pandas as pd

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
train = pd.read_csv(os.path.join(data_directory, 'fashion_train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'fashion_valid_split.csv'))
train['filler_column'] = 'a'
train_output = train[['Category', 'filler_column', 'translated_title']]
valid_output = valid[['Category', 'translated_title']]
train_output.to_csv(os.path.join(data_directory, 'fashion_train.tsv'),
                    sep='\t', index=True, header=False)
valid_output.to_csv(os.path.join(data_directory, 'fashion_valid.tsv'),
                    sep='\t', index=False, header=['id', 'sentence'])
