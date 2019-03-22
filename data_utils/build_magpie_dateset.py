import os
import pandas as pd

"""Magpie is not used in the final model so this script can be ignored"""

psychic_learners_dir = os.path.split(os.getcwd())[0]
data_directory = os.path.join(psychic_learners_dir, 'data')
big_categories = ['beauty', 'fashion', 'mobile']
for big_category in big_categories:
    train = pd.read_csv(os.path.join(data_directory, f'{big_category}_train_split.csv'))
    valid = pd.read_csv(os.path.join(data_directory, f'{big_category}_valid_split.csv'))
    test = pd.read_csv(os.path.join(data_directory, f'{big_category}_test_split.csv'))
    train_output_dir = os.path.join(data_directory, 'magpie', big_category, 'train')
    valid_output_dir = os.path.join(data_directory, 'magpie', big_category, 'valid')
    test_output_dir = os.path.join(data_directory, 'magpie', big_category, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(valid_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    for row in train.itertuples():
        itemid = row[1]
        title = row[2]
        category = row[3]
        with open(os.path.join(train_output_dir, f'{itemid}.txt'), 'w') as f:
            f.write(title)
        with open(os.path.join(train_output_dir, f'{itemid}.lab'), 'w') as f:
            f.write(str(category))
    
    for row in valid.itertuples():
        itemid = row[1]
        title = row[2]
        category = row[3]
        with open(os.path.join(valid_output_dir, f'{itemid}.txt'), 'w') as f:
            f.write(title)
        with open(os.path.join(valid_output_dir, f'{itemid}.lab'), 'w') as f:
            f.write(str(category))

    for row in test.itertuples():
        itemid = row[1]
        title = row[2]
        with open(os.path.join(test_output_dir, f'{itemid}.txt'), 'w') as f:
            f.write(title)
    

        
