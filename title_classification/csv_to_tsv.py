import os
import pandas as pd

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
for big_category in ['beauty', 'fashion', 'mobile']:
    train = pd.read_csv(os.path.join(data_directory, f'{big_category}_train_split.csv'))
    valid = pd.read_csv(os.path.join(data_directory, f'{big_category}_valid_split.csv'))
    train['filler_column'] = 'a'
    valid['filler_column'] = 'a'
    train_output = train[['Category', 'filler_column', 'title']]
    valid_output = valid[['Category', 'filler_column', 'title']]
    #test_output = valid[['Category', 'title']]
    output_dir = os.path.join(data_directory, 'tsvs', 'unclean_untranslated', big_category)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    train_output.to_csv(os.path.join(output_dir, 'train.tsv'),
                        sep='\t', index=True, header=False)
    valid_output.to_csv(os.path.join(output_dir, 'dev.tsv'),
                        sep='\t', index=True, header=False)
    #test_output.to_csv(os.path.join(data_directory, f'{big_category}_valid.tsv'),
    #                    sep='\t', index=False, header=['id', 'sentence'])
