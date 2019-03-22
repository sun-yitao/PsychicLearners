import os
from glob import glob
import cv2
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

# This script downscales images to 240x240 to make it faster to transmit to the cloud gpu for training
psychic_learners_dir = os.path.split(os.getcwd())[0]
INPUT_DIRECTORY_NAME = 'test'
small_categories = [[1, 0, 7, 14, 2, 8, 5, 4, 13, 11, 15, 3, 10, 9, 6, 16, 12],
                    [23, 27, 18, 20, 24, 22, 19, 26, 25, 29, 28, 17, 21, 30],
                    [35, 53, 40, 39, 52, 45, 31, 51, 49, 56, 38, 34, 46, 33, 
                     57, 37, 55, 32, 42, 44, 50, 36, 43, 54, 41, 47, 48]]


def resize_image(input_path):
    im = cv2.imread(input_path)
    small_im = cv2.resize(im, (240, 240), interpolation=cv2.INTER_AREA)
    category, filename = os.path.split(input_path)
    category = os.path.split(category)[-1]
    cv2.imwrite(os.path.join(output_directory,
                                category, filename), small_im)

def resize_test_image(input_path):
    im = cv2.imread(input_path)
    small_im = cv2.resize(im, (240, 240), interpolation=cv2.INTER_AREA)
    _, filename = os.path.split(input_path)
    cv2.imwrite(os.path.join(output_directory, filename), small_im)

"""
for n, big_category in enumerate(['beauty', 'fashion', 'mobile']):
    input_directory = os.path.join(psychic_learners_dir, 'data', 'image', INPUT_DIRECTORY_NAME, big_category)
    output_directory = os.path.join(psychic_learners_dir, 'data', 'image', INPUT_DIRECTORY_NAME + '_240x240', big_category)
    
    if not os.path.isdir(output_directory):
        for i in small_categories[n]: 
            os.makedirs(os.path.join(output_directory, str(i)), exist_ok=True)
    

    imagesList = glob(os.path.join(input_directory, '**', '*.jpg'), recursive=True)
    pool = ThreadPool(6)
    pool.map(resize_image, imagesList)"""

input_directory = os.path.join(psychic_learners_dir, 'data', 'image', INPUT_DIRECTORY_NAME)
output_directory = os.path.join(psychic_learners_dir, 'data', 'image', INPUT_DIRECTORY_NAME + '_240x240')
os.makedirs(output_directory, exist_ok=True)
imagesList = glob(os.path.join(input_directory, '*.jpg'))
pool = ThreadPool(6)
pool.map(resize_test_image, imagesList)
