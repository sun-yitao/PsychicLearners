import os
from glob import glob
import pandas as pd


psychic_learners_dir = os.path.split(os.getcwd())[0]
image_folder = os.path.join(psychic_learners_dir, 'data', 'image', 'v1_train_nodups_240x240')
train_df = pd.read_csv(os.path.join(psychic_learners_dir, 'data', 'train_split.csv'))

for big_category in glob(os.path.join(image_folder, '*')):
    if not os.path.isdir(big_category):
        continue
    for small_category in glob(os.path.join(big_category, '*')):
        if not os.path.isdir(small_category):
            continue
        category = int(os.path.split(small_category)[-1])
        df = train_df.loc[train_df['Category'] == category]
        df['image_names'] = df['abs_image_path'].map(lambda x: os.path.split(x)[-1])
        print(df['image_names'])
        break
        for image in glob(os.path.join(small_category, '*.jpg')):
            image_name = os.path.split(image)[-1]
            if image_name not in df['image_names']:
                print('{} placed in wrong category {}'.format(image_name, category))
            else:
                print('correct')


