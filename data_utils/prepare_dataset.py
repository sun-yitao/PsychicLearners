import tarfile
import os
import time
import json
from glob import glob
from shutil import copy, move

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from google.cloud import translate
from tqdm import tqdm
from spellchecker import SpellChecker
import multiprocessing as mp

psychic_learners_dir = os.path.split(os.getcwd())[0]
data_directory = os.path.join(psychic_learners_dir, 'data')
if not os.path.exists(data_directory):
    os.makedirs(data_directory, exist_ok=True)
image_directory = os.path.join(data_directory, 'image')
output_dir = os.path.join(image_directory, 'original')
train_dir = os.path.join(image_directory, 'v1_train_240x240')
valid_dir = os.path.join(image_directory, 'valid_240x240')

def _image_filename_path_writer(filename):
    abs_path = os.path.join(output_dir, filename)
    if not abs_path.endswith('.jpg'):
        abs_path += '.jpg'
    return abs_path

train_df = pd.read_csv(os.path.join(data_directory, 'train.csv'))
train_df['abs_image_path'] = train_df['image_path'].map(_image_filename_path_writer)
test = pd.read_csv(os.path.join(data_directory, 'test.csv'))
test['abs_image_path'] = test['image_path'].map( _image_filename_path_writer)
# Use StratifiedShuffleSplit to split images and train.csv into training and validation sets
# Stratification ensures that both splits do not have missing categories
train, valid = train_test_split(train_df,
                                stratify=train_df['Category'],
                                test_size=0.2, random_state=42)

def extract_tar_images():
    # extract tarfiles in data directory to image directory
    tarfiles = glob(os.path.join(data_directory, '*.tar.gz'))
    for t in tqdm(tarfiles):
        tf = tarfile.open(t)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        tf.extractall(path=output_dir)

translations_mapping = {}
word_to_lang = {}
count_vect = CountVectorizer(analyzer='word', strip_accents='unicode', 
                             token_pattern=r'\b[^\d\W]{3,}\b')  # match words with 3 or more letters
titles = pd.concat([test['title'], train_df['title']])
v = count_vect.fit(titles)

def get_translations_dict():
    # use google translate api to get a dict of translations mapping and save it for future use
    translate_client = translate.Client()
    target = 'en' # translate all to english
    for word, count in tqdm(v.vocabulary_.items()):
        #context_string = titles[titles.str.contains(word)].head(1).values[0]
        try:
            result = translate_client.detect_language(word)
        except:
            print('Error detecting {}'.format(word))
            continue
        detected_language = result['language']
        if detected_language != target:
            try:
                translation = translate_client.translate(word, source_language=detected_language, target_language=target)
            except:
                print('Error translating {}'.format(word))
                continue
            word_to_lang[word] = detected_language
            if word != translation['translatedText']:
                translations_mapping[word] = translation['translatedText']
                #print(detected_language, word, translation['translatedText'])
    with open('translations_mapping.json', 'w') as file:
        file.write(json.dumps(translations_mapping))
    with open('word_to_lang.json', 'w') as file:
        file.write(json.dumps(word_to_lang))

def translate_sentence(sentence):
    words = sentence.split(' ')
    for n, word in enumerate(words):
        if word in translations_mapping:
            words[n] = translations_mapping[word]
    return ' '.join(words)

def translate_df(dataframe):
    #translate train, valid and test titles to english on a new column
    translated_df = dataframe.copy()
    translated_df['translated_title'] = translated_df['title'].map(translate_sentence)
    return translated_df

weird_words = set()
def detect_weird_sentence(sentence):
    words = sentence.split(' ')
    for word in words:
        if word not in v.vocabulary_.keys():
            weird_words.add(word)

def get_weird_sentences():
    titles.map(detect_weird_sentence)
    weird_words = list(weird_words)
    weird_words = [word + '\n' for word in weird_words]
    with open('weird_words.txt', 'w') as f:
        f.writelines(weird_words)

spell = SpellChecker()
spelling_errors = set()
correct_words = set()
wrong_words = set()
def detect_spelling_mistake(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in wrong_words or word in correct_words or word in translations_mapping.keys(): # word already checked
            continue
        correction = spell.correction(word)
        if correction != word: # if word is not english and spelt wrongly
            spelling_errors.add((word, correction))
            wrong_words.add(word)
            print(word)
        else: # if word is not english or is english and spelt correctly
            correct_words.add(word)


def process(df):
    res = df.map(detect_spelling_mistake)
    return res

def get_spelling_mistakes():
    cpus = mp.cpu_count()
    p = mp.Pool(cpus)
    p.map(process, np.array_split(titles, cpus))
    p.close()
    p.join()
    global spelling_errors
    spelling_errors = list(spelling_errors)
    spelling_errors = [word + spell.correction(word) + '\n' for word in spelling_errors]
    with open('spelling_errors.txt', 'w') as f:
        f.writelines(spelling_errors)

def combine_spelling_and_weird_txt():
    #combine the 2 files into a single mapping
    misspelt_mappings = {}
    misspelt_words = set()
    with open('spelling_errors.txt', 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            misspelt, correction = line.split(' ')
            misspelt_mappings[misspelt] = correction
            misspelt_words.add(misspelt)
    with open('weird_words.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]
        weird_words = set(lines)
    diff = weird_words.difference(misspelt_words) # get words in weird words that are not in misspelt words
    for word in diff:
        misspelt_mappings[word] = 'WEIRD'
    with open('misspelt_and_weird_mappings.json', 'w') as file:
        file.write(json.dumps(misspelt_mappings))
        
        

def clean_and_translate_df(dataframe):
    translated_df = dataframe.copy()
    translated_df['translated_title'] = translated_df['title'].map(translate_sentence)
    translated_df['clean_translated_title'] = translated_df['translated_title'].map(clean_sentence)
    return translated_df

def clean_df(dataframe):
    cleaned_df = dataframe.copy()
    clean_df['cleaned_title'] = cleaned_df['title'].map(clean_sentence)
    return cleaned_df

def clean_sentence(sentence):
    words = sentence.split(' ')
    for n, word in enumerate(words):
        if word in misspelt_mappings:
            words[n] = misspelt_mappings[word]
    return ' '.join(words)

def make_csvs():
    train.to_csv(os.path.join(data_directory, 'train_split.csv'), index=False)
    valid.to_csv(os.path.join(data_directory, 'valid_split.csv'), index=False)
    test.to_csv(os.path.join(data_directory, 'test_split.csv'), index=False)

    beauty_train = train.loc[(train['Category'] >= 0) & (train['Category'] <= 16)]
    beauty_valid = valid.loc[(valid['Category'] >= 0) & (valid['Category'] <= 16)]
    beauty_train.to_csv(os.path.join(data_directory, 'beauty_train_split.csv'), index=False)
    beauty_valid.to_csv(os.path.join(data_directory, 'beauty_valid_split.csv'), index=False)

    fashion_train = train.loc[(train['Category'] >= 17) & (train['Category'] <= 30)]
    fashion_valid = valid.loc[(valid['Category'] >= 17) & (valid['Category'] <= 30)]
    fashion_train.to_csv(os.path.join(data_directory, 'fashion_train_split.csv'), index=False)
    fashion_valid.to_csv(os.path.join(data_directory, 'fashion_valid_split.csv'), index=False)

    mobile_train = train.loc[(train['Category'] >= 31) & (train['Category'] <= 57)]
    mobile_valid = valid.loc[(valid['Category'] >= 31) & (valid['Category'] <= 57)]
    mobile_train.to_csv(os.path.join(data_directory, 'mobile_train_split.csv'), index=False)
    mobile_valid.to_csv(os.path.join(data_directory, 'mobile_valid_split.csv'), index=False)

def copy_images_to_image_dir():
    n_categories = train_df['Category'].nunique()
    # create directory structure such that every category is represented by a folder
    # this will make balancing and augmenting the dataset easier
    for i in range(n_categories):
        os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, str(i)), exist_ok=True)
    os.makedirs(os.path.join(image_directory, 'test'), exist_ok=True)

    # copy the images to the respective categories, if low on disk space change to shutil.move
    # not well optimised takes forever to run
    print('Copying training images')
    for row in tqdm(train.itertuples()):
        copy(row[5], os.path.join(train_dir, str(row[3])))
    print('Copying validation images')
    for row in tqdm(valid.itertuples()):
        copy(row[5], os.path.join(valid_dir, str(row[3])))
    print('Copying test images')
    for row in tqdm(test.itertuples()):
        copy(row[4], os.path.join(image_directory, 'test'))

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

def check_copied_images_correct():
    # check that all the images are present without duplicates
    images = glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(valid_dir, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(image_directory, 'test', '*.jpg'))
    original_images = glob(os.path.join(output_dir, '**', '*.jpg'), recursive=True)
    images = [os.path.split(filename)[-1] for filename in images]
    original_images = [os.path.split(filename)[-1] for filename in original_images]
    assert sorted(images) == sorted(original_images)

if __name__ == '__main__':
    #extract_tar_images() 
    #get_translations_dict()  # uncomment this to get a new translation mapping else just load the one already built
    with open('translations_mapping.json', 'r') as f:
        translations_mapping = json.load(f)
    #with open('misspelt_and_weird_mappings.json', 'r') as f:
    #    misspelt_mappings = json.load(f)
    """train = clean_and_translate_df(train)
    valid = clean_and_translate_df(valid)
    test = clean_and_translate_df(test)
    make_csvs()"""
    get_spelling_mistakes()
    #copy_images_to_image_dir()
    #check_copied_images_correct()
