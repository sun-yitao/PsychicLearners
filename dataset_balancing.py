import os
import pandas as pd
from shutil import copy

EXAMPLES_PER_CLASS = 10000
train_split_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train_split.csv'))
train_split_df.sort_values(by=['Category'], inplace=True)
value_counts = train_split_df['Category'].value_counts()
balanced_df = pd.DataFrame()

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
    balanced_df = balanced_df.append(class_examples_df, ignore_index=True)
    
print(balanced_df['Category'].value_counts())
balanced_df.to_csv(os.path.join(os.getcwd(), 'data',
                                'train_split_balanced.csv'), index=False)
