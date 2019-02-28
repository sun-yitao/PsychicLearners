import os
from glob import glob
import cv2
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

#input_directory = os.path.join(os.getcwd(), 'data', 'image', 'v1_train')
#output_directory = os.path.join(os.getcwd(), 'data', 'image', 'v1_train_240x240')
input_directory = os.path.join(os.getcwd(), 'data', 'image', 'v1_train_balanced')
output_directory = os.path.join(os.getcwd(), 'data', 'image', 'v1_train_balanced_240x240')
if not os.path.isdir(output_directory):
    for i in range(58): # categories 0-57
        os.makedirs(os.path.join(output_directory, str(i)), exist_ok=True)


def resize_image(input_path):
    im = cv2.imread(input_path)
    small_im = cv2.resize(im, (240, 240), interpolation=cv2.INTER_AREA)
    category, filename = os.path.split(input_path)
    category = os.path.split(category)[-1]
    cv2.imwrite(os.path.join(output_directory, category, filename), small_im)

imagesList = glob(os.path.join(input_directory, '**', '*.jpg'), recursive=True)
pool = ThreadPool(6)
pool.map(resize_image, imagesList)
