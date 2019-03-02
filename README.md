# PsychicLearners

NDSC 2019

### Preparing Dataset

Put all csv and tar.gz files in data directory
from Psychic Learners directory run

```
python prepare_dataset.py
```

This will untar the tarfiles, shuffle the examples in train.csv deterministically to produce train_split.csv and valid_split.csv with a 80/20 split. These 2 new csvs will contain a new column of absolute image paths to the original image directory. Finally the script will copy images from the original directory to a v1 image image directory sorted by category (Possibly redundant now can remove in future as v1 is only for testing). Takes a few hours to run as untar and copying is slow.

Then run `python dataset_balancing.py` 
This will create a new train_split_balanced.csv which contains 3000 examples per class, downsampling/upsampling if needed. Also moves images from original directory to v1_train_balanced that contains 10000 images per class (takes very long to run). Note that this will degrade accuracy, but is needed to train and test models faster. Without downsampling the dataset one model would take 1 week on a P6000 gpu to train.



### Exploratory Data Analysis

Please refer to EDA.ipynb and tsne.ipynb



### Title Classification

???



### Image Classification

Plan to do ensembling with a pretrained nasnet and a retrained SEnet ResNeXt.
