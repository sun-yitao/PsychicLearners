import os
from glob import glob

from image_match.goldberg import ImageSignature
from elasticsearch import Elasticsearch
from image_match.elasticsearch_driver import SignatureES
# Need to start elastic search $elasticsearch on osx, $sudo service elasticsearch start on ubuntu

psychic_learners_dir = os.path.split(os.getcwd())[0]
image_directory = os.path.join(psychic_learners_dir, 'data', 'image', 'train_v1')
category_directories = glob(os.path.join(image_directory, '*'))
for category_directory in category_directories:
    image_filenames = glob(os.path.join(category_directory, '*.jpg'))
    es = Elasticsearch()
    ses = SignatureES(es)
    for image_filename in image_filenames:
        ses.add_image(image_filename)
    for image_filename in image_filenames:
        ses.search_image(image_filename)

