import tarfile
import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copy, move

psychic_learners_dir = os.path.split(os.getcwd())[0]
data_directory = os.path.join(psychic_learners_dir, 'data')
if not os.path.exists(data_directory):
    os.makedirs(data_directory, exist_ok=True)
image_directory = os.path.join(data_directory, 'image')
output_dir = os.path.join(image_directory, 'original')
train_dir = os.path.join(image_directory, 'v1_train_240x240')
valid_dir = os.path.join(image_directory, 'valid_240x240')

"""
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
    os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, str(i)), exist_ok=True)
os.makedirs(os.path.join(image_directory, 'test'), exist_ok=True)

# copy the images to the respective categories, if low on disk space change to shutil.move
# not well optimised takes forever to run
print('Copying training images')
for row in train.itertuples():
    copy(row[5], os.path.join(train_dir, str(row[3])))
print('Copying validation images')
for row in valid.itertuples():
    copy(row[5], os.path.join(valid_dir, str(row[3])))
print('Copying test images')
for row in test.itertuples():
    copy(row[4], os.path.join(image_directory, 'test'))"""

# This was added after it was officially confirmed that the 3 big categories can be used in prediction
# in order to sort the category folders into big categories and apply a separate classifier to each big category
train_category_directories = glob(os.path.join(train_dir, '*'))
train_category_directories = [dir for dir in train_category_directories if os.path.isdir(dir)]
valid_category_directories = glob(os.path.join(valid_dir, '*'))
valid_category_directories = [dir for dir in valid_category_directories if os.path.isdir(dir)]
mobile_categories = [35, 53, 40, 39, 52, 45, 31, 51, 49, 56, 38,
                     34, 46, 33, 57, 37, 55, 32, 42, 44, 50, 36, 43, 54, 41, 47, 48]
fashion_categories = [23, 27, 18, 20, 24, 22, 19, 26, 25, 29, 28, 17, 21, 30]
beauty_categories = [1, 0, 7, 14, 2, 8, 5, 4, 13, 11, 15, 3, 10, 9, 6, 16, 12]

def move_directory_to_big_category(small_category_directory):
    dst, category_number = os.path.split(small_category_directory)
    category_number = int(category_number)
    if category_number in mobile_categories:
        dst = os.path.join(dst, 'mobile')
        move(small_category_directory, dst)
    elif category_number in fashion_categories:
        dst = os.path.join(dst, 'fashion')
        move(small_category_directory, dst)
    elif category_number in beauty_categories:
        dst = os.path.join(dst, 'beauty')
        move(small_category_directory, dst)

for big_category in ['beauty', 'fashion', 'mobile']:
    os.makedirs(os.path.join(train_dir, big_category), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, big_category), exist_ok=True)

for train_cat_dir in train_category_directories:
    move_directory_to_big_category(train_cat_dir)
for valid_cat_dir in valid_category_directories:
    move_directory_to_big_category(valid_cat_dir)

# check that all the images are present without duplicates
images = glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True) + \
         glob(os.path.join(valid_dir, '**', '*.jpg'), recursive=True) + \
         glob(os.path.join(image_directory, 'test', '*.jpg'))
original_images = glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True)
images = [os.path.split(filename)[-1] for filename in images]
original_images = [os.path.split(filename)[-1] for filename in original_images]
assert sorted(images) == sorted(original_images)
