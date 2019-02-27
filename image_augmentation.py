import os
from os.path import join, split
from glob import glob
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

# This script will augment each image in place, overwriting the original.
train_dir = join(os.getcwd(), 'data', 'image', 'v1_train_balanced')
train_folders = glob(join(train_dir, '*'))

# array of paths to image directories to augment
image_directories = glob('/Users/sunyitao/Documents/XRVision/STE/dataset/v4/train/*')
image_extension = '.jpg'
augmentation_factor = 2  # factor of number of original images to generate

def augment_images(np_img_array, img_dir):
    seq = iaa.Sequential([
        iaa.Sometimes(0.8, 
            iaa.CropAndPad(
                percent=(0, 0.2),
                pad_mode=["edge", "reflect"],
            )),
        iaa.Sometimes(0.35, iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2)),
        iaa.Sometimes(0.35, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25))),
        iaa.Sometimes(0.35, iaa.OneOf([
            iaa.CoarseDropout((0.15, 0.2), size_percent=(0.001, 0.02), per_channel=0.1),
            iaa.CoarseSaltAndPepper((0.15, 0.2), size_percent=(0.001, 0.02)),
            iaa.Superpixels(p_replace=(0.15, 0.2), n_segments=(128, 256))
            ])
        )],
        random_order=True)

    images_aug = seq.augment_images(np_img_array)
    for image in images_aug:
        global image_num
        image_num += 1
        im = Image.fromarray(image)
        im.save(join(image_dir, 'aug_{}_img_{}{}'.format(split(img_dir)[-1],
                                                         image_num,
                                                         image_extension)))

for image_dir in image_directories:
    image_list = sorted(glob(join(image_dir, '*' + image_extension)))
    np_img_array = []

    for image in image_list:
        np_image = np.array(Image.open(image))
        np_img_array.append(np_image)
    image_num = 0

    for cycle in range(augmentation_factor):
        print('Image Directory: {} Cycle: {}'.format(image_dir, cycle))
        augment_images(np_img_array, image_dir)

