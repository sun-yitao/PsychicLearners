import tarfile
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy

cwd = os.getcwd() # this needs to be performed in PsychicLearners root directory
data_directory = os.path.join(cwd, 'data')
image_directory = os.path.join(data_directory, 'image')
output_dir = os.path.join(image_directory, 'original')
# extract tarfiles in data directory to image directory

tarfiles = glob(os.path.join(data_directory, '*.tar.gz'))
for t in tarfiles:
    tf = tarfile.open(t)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    tf.extractall(path=output_dir)


# Use StratifiedShuffleSplit to split images and train.csv into training and validation sets
# Stratification ensures that both splits do not have missing categories
def image_filename_path_writer(filename):
    abs_path = os.path.join(output_dir, filename)
    if not abs_path.endswith('.jpg'):
        abs_path += '.jpg'
    return abs_path

train_df = pd.read_csv(os.path.join(data_directory, 'train.csv'))
train_df['abs_image_path'] = train_df['image_path'].map(image_filename_path_writer)
test = pd.read_csv(os.path.join(data_directory, 'test.csv'))
test['abs_image_path'] = test['image_path'].map(image_filename_path_writer)
train, valid = train_test_split(train_df,
                                stratify=train_df['Category'],
                                test_size=0.2, random_state=42)
train.to_csv(os.path.join(data_directory, 'train_split.csv'), index=False)
valid.to_csv(os.path.join(data_directory, 'valid_split.csv'), index=False)

n_categories = train_df['Category'].nunique()
# create directory structure such that every category is represented by a folder
# this will make balancing the dataset easier
for i in range(n_categories):
    os.makedirs(os.path.join(image_directory, 'v1', 'train', str(i)), exist_ok=True)
    os.makedirs(os.path.join(image_directory, 'v1', 'valid', str(i)), exist_ok=True)
os.makedirs(os.path.join(image_directory, 'test'), exist_ok=True)

# copy the images to the respective categories, if low on disk space change to shutil.move
# not well optimised takes forever to run
print('Copying training images')
for row in train.itertuples():
    copy(row[5], os.path.join(image_directory, 'v1', 'train', str(row[3])))
print('Copying validation images')
for row in valid.itertuples():
    copy(row[5], os.path.join(image_directory, 'v1', 'valid', str(row[3])))
print('Copying test images')
for row in test.itertuples():
    copy(row[4], os.path.join(image_directory, 'test'))

# check that all the images are present without duplicates
images = glob(os.path.join(image_directory, 'v1', '**', '*.jpg'), recursive=True) + \
         glob(os.path.join(image_directory, 'test', '*.jpg'))
original_images = glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True)
images = [os.path.split(filename)[-1] for filename in images]
original_images = [os.path.split(filename)[-1] for filename in original_images]
print(len(images))
print(len(original_images))
assert sorted(images) == sorted(original_images)
