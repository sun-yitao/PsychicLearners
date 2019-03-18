import os
import pandas as pd

data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
for big_category in ['beauty', 'fashion', 'mobile']:
    #train = pd.read_csv(os.path.join(data_directory, f'{big_category}_train_split.csv'))
    valid = pd.read_csv(os.path.join(data_directory, f'{big_category}_valid_split.csv'))
    test = pd.read_csv(os.path.join(
        data_directory, f'{big_category}_test_split.csv'))
    #train['filler_column'] = 'a'
    #valid['filler_column'] = 'a'
    #train_output = train[['Category', 'filler_column', 'title']]
    valid_output = valid[['itemid', 'title']]
    test_output = test[['itemid', 'title']]
    output_dir = os.path.join(data_directory, 'tsvs', 'for_inference', big_category)
    
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    #train_output.to_csv(os.path.join(output_dir, 'train.tsv'),
    #                    sep='\t', index=True, header=False)
    valid_output.to_csv(os.path.join(output_dir, 'valid', 'test.tsv'),
                        sep='\t', index=False, header=['id', 'sentence'])
    test_output.to_csv(os.path.join(output_dir, 'test', 'test.tsv'),
                        sep='\t', index=False, header=['id', 'sentence'])
