import os
from os.path import join, split
from glob import glob
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

# This script will augment all images in a directory to a certain multiple, deleting the original
# if you wish to keep original please duplicate before running
train_dir = join(os.path.split(os.getcwd())[0], 'data', 'image',
                 'v1_train_undersampled_3k_240x240_augmented')
image_directories = glob(join(train_dir, '*'))
image_extension = '.jpg'
augmentation_factor = 1  # factor of number of original images to generate

ia.seed(42) # to produce reproducible augmentations
def augment_images(np_img_array, img_dir, img_list):
    seq = iaa.Sequential([
        iaa.Sometimes(0.8, 
            iaa.CropAndPad(
                percent=(0.1, 0.3),
                pad_mode=["edge", "reflect"],
            )),
        iaa.Sometimes(0.35, iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.WithChannels(0, iaa.Add((10, 50)))
        )),
        iaa.Sometimes(0.35, iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
        iaa.Sometimes(0.35, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25))),
        iaa.Sometimes(0.35, iaa.OneOf([
            iaa.CoarseDropout((0.15, 0.2), size_percent=(0.001, 0.02), per_channel=0.1),
            iaa.CoarseSaltAndPepper((0.15, 0.2), size_percent=(0.001, 0.02)),
            #iaa.Superpixels(p_replace=(0.15, 0.2), n_segments=(128, 256))
            ])
        )], random_order=True)
    images_aug = seq.augment_images(np_img_array)
    for image, filepath in zip(images_aug, image_list):
        global image_num
        image_num += 1
        im = Image.fromarray(image)
        new_filename = split(filepath)[-1]
        new_filename.replace(image_extension, '')
        new_filename = new_filename + str(image_num) + image_extension
        im.save(join(image_dir, new_filename))

def remove_image(path):
    try:
        os.remove(path)
    except:
        print('Error removing {}'.format(path))

for image_dir in image_directories:
    image_list = sorted(glob(join(image_dir, '*' + image_extension)))
    np_img_array = []

    for image in image_list:
        np_image = np.array(Image.open(image))
        np_img_array.append(np_image)
    image_num = 0
    #delete original images
    pool = ThreadPool(6)
    pool.map(remove_image, image_list)

    for cycle in range(augmentation_factor):
        print('Image Directory: {} Cycle: {}'.format(image_dir, cycle))
        augment_images(np_img_array, image_dir, image_list)

