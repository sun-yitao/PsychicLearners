import os
import pandas as pd
from shutil import copy

"""This script is not used any of the final models and can be ignored"""

EXAMPLES_PER_CLASS = 3000
psychic_learners_dir = os.path.split(os.getcwd())[0]
train_split_df = pd.read_csv(os.path.join(psychic_learners_dir,
                             'data', 'train_split.csv'))
train_split_df.sort_values(by=['Category'], inplace=True)
value_counts = train_split_df['Category'].value_counts()
balanced_df = pd.DataFrame()

def remove_extension(path):
    new_path = path[:-4]
    return new_path

# make each category the same num of examples
for category in range(train_split_df['Category'].nunique()):
    n_examples = value_counts[category]
    class_examples_df = train_split_df.loc[train_split_df['Category'] == category]

    if n_examples > EXAMPLES_PER_CLASS:
        print('Undersampling category {} from {} to {}'.format(category, n_examples, EXAMPLES_PER_CLASS))
        class_examples_df = class_examples_df.sample(n=EXAMPLES_PER_CLASS, random_state=category)
    elif n_examples < EXAMPLES_PER_CLASS:
        print('Oversampling category {} from {} to {}'.format(category, n_examples, EXAMPLES_PER_CLASS))
        multiplier = int(EXAMPLES_PER_CLASS / n_examples) + 1
        class_examples_df = pd.concat([class_examples_df] * multiplier, ignore_index=True).head(EXAMPLES_PER_CLASS)
        # This will make duplicate images have unique paths
        class_examples_df['abs_image_path'] = class_examples_df['abs_image_path'].map(remove_extension)
        class_examples_df['abs_image_path'] = class_examples_df['abs_image_path'] + class_examples_df.index.astype(str) + '.jpg'
    balanced_df = balanced_df.append(class_examples_df, ignore_index=True)

# shuffle rows
balanced_df = balanced_df.sample(frac=1, random_state=42)
print(balanced_df['Category'].value_counts())
balanced_df.to_csv(os.path.join(psychic_learners_dir, 'data',
                                'train_split_balanced_3k.csv'), index=False)

# copy images to make new dataset
for i in range(train_split_df['Category'].nunique()):
    os.makedirs(os.path.join(psychic_learners_dir, 'data', 'image',
                             'v1_train_balanced_3k', str(i)), exist_ok=True)
for row in balanced_df.itertuples():
    dest = os.path.join(psychic_learners_dir, 'data', 'image',
                        'v1_train_balanced_3k', str(row[3]),
                        os.path.split(row[5])[-1])
    src = os.path.join(psychic_learners_dir, 'data', 'image', 'original', str(row[4]))
    if not src.endswith('.jpg'):
        src += '.jpg'
    copy(src, dest)

"""
Oversampling category 0 from 3018 to 10000
Undersampling category 1 from 22936 to 10000
Oversampling category 2 from 9235 to 10000
Undersampling category 3 from 65000 to 10000
Undersampling category 4 from 34150 to 10000
Undersampling category 5 from 44223 to 10000
Oversampling category 6 from 1638 to 10000
Oversampling category 7 from 9334 to 10000
Oversampling category 8 from 4838 to 10000
Oversampling category 9 from 6485 to 10000
Oversampling category 10 from 862 to 10000
Oversampling category 11 from 3286 to 10000
Undersampling category 12 from 17426 to 10000
Oversampling category 13 from 2386 to 10000
Oversampling category 14 from 2188 to 10000
Oversampling category 15 from 479 to 10000
Oversampling category 16 from 1782 to 10000
Oversampling category 17 from 2212 to 10000
Undersampling category 18 from 45279 to 10000
Undersampling category 19 from 10794 to 10000
Undersampling category 20 from 15982 to 10000
Oversampling category 21 from 8386 to 10000
Undersampling category 22 from 12180 to 10000
Oversampling category 23 from 1337 to 10000
Oversampling category 24 from 3401 to 10000
Undersampling category 25 from 27138 to 10000
Undersampling category 26 from 27076 to 10000
Undersampling category 27 from 12959 to 10000
Oversampling category 28 from 5194 to 10000
Oversampling category 29 from 2670 to 10000
Oversampling category 30 from 1153 to 10000
Undersampling category 31 from 22269 to 10000
Undersampling category 32 from 23803 to 10000
Oversampling category 33 from 3857 to 10000
Undersampling category 34 from 11774 to 10000
Undersampling category 35 from 24472 to 10000
Oversampling category 36 from 822 to 10000
Oversampling category 37 from 1818 to 10000
Oversampling category 38 from 3762 to 10000
Oversampling category 39 from 558 to 10000
Oversampling category 40 from 262 to 10000
Undersampling category 41 from 15450 to 10000
Oversampling category 42 from 8402 to 10000
Oversampling category 43 from 4918 to 10000
Oversampling category 44 from 736 to 10000
Oversampling category 45 from 1756 to 10000
Oversampling category 46 from 547 to 10000
Oversampling category 47 from 756 to 10000
Oversampling category 48 from 333 to 10000
Oversampling category 49 from 458 to 10000
Oversampling category 50 from 226 to 10000
Oversampling category 51 from 323 to 10000
Oversampling category 52 from 94 to 10000
Oversampling category 53 from 334 to 10000
Oversampling category 54 from 240 to 10000
Oversampling category 55 from 121 to 10000
Oversampling category 56 from 136 to 10000
Oversampling category 57 from 38 to 10000
"""
