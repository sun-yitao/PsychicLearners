# PsychicLearners

NDSC 2019

## Overview

Our algorithm consists of a number of level 1 learners and a meta learner trained on the predicted probabilities of the level 1 algorithms.  Due to a lack of time, we were unable to do cross-validation on many of the more computationally expensive algorithms and a proper parameter search on the best configuration of the level 1 algorithms that leads to the best accuracy of the ensemble as a whole. We also apologize for the poor code quality as it was written in a hurry.

#### Level 1 algorithms:

From [https://github.com/TobiasLee/Text-Classification]

    1) Adversarial Training Methods for Semi-Supervised Text Classification [https://arxiv.org/abs/1605.07725]

    2) Attention-Based Bidirectional Long Short-Term Memory Networks [http://www.aclweb.org/anthology/P16-2034]

    3) Independently Recurrent Neural Network [https://arxiv.org/abs/1803.04831]

    4) Multihead Attention Module [https://arxiv.org/abs/1706.03762]



From [https://github.com/dongjun-Lee/text-classification-models-tf]

    5) Word-level CNN [https://arxiv.org/abs/1408.5882]

    6) Character-level CNN [https://arxiv.org/abs/1509.01626]

    7) Word-level Bi-RNN [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761987.pdf]

    8) RCNN [https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552]



9) Fasttext on titles [https://arxiv.org/abs/1607.01759]

10) Fasttext on extracted text with pytesseract [https://github.com/madmaze/pytesseract]

11) BERT Finetuning [https://arxiv.org/abs/1810.04805https://arxiv.org/abs/1810.04805] and  [https://github.com/google-research/bert]

12) Image classification: We use a SEResNext50 for fashion images and a pretrained Inception Resnet V2 for mobile and beauty images, with random erasing augmentation. 



#### Meta Learner:

XGBoost on probabilities of level 1 models



## Steps to reproduce meta model results:

Download models and predicted probabilties from google drive

Put them in PsychicLearners/data/probabilities and PsychicLearners/data/keras_checkpoints (keras checkpoints is misnamed, it actually just refers to all checkpoints keras or non-keras)

Uncomment  the predict_all_xgb() function and run

```
cd ensemble
python stack_ensemble.py
```



## Steps to reproduce level 1 models

### First prepare dataset

1) Put all the original csvs and images into PsychicLearners/data and PsychicLearners/data/image/original

2) Modify data_utils/prepare_dataset.py.  The script will do a 80/20 stratified split of train.csv and the dataframes will be saved in variables train, valid and test. We didn't use most of the functions in here as they did not seem to help improve performance.

3) To get text extractions from image run get_text_extractions_parallel(dataframe)

4) make_csvs() will split train, valid and test dataframe into the 3 big categories (beauty, fashion and mobile) and persist to disk

5) copy_images_to_directory() moves the images into folders 0-57 such that in can be read by keras image generator flow_from_directory. We originally intended to use this format to make it easier to do class balancing and offline augmentation, but in the end we did train-time augmentation and used class weights to avoid increasing the size of the already large dataset. This function was badly written as we did originally did not know the 3 big categories given can be used for classification. 

For the rest of the cleaning and translating functions, we found that they did not help improve performance and did not use them in the end.



### For the models under [https://github.com/TobiasLee/Text-Classification]:

Modify the script under 

```python
data_dir = 'path/to/csv' #directory containing csv files named {BIG_CATEGORY}_{SUBSET}_split.csv
    BIG_CATEGORY = 'beauty' #beauty, fashion or mobile
    N_CLASS=17 # 17, 14, 27 respectively
    x_train, y_train = load_data(os.path.join(data_dir, f'{BIG_CATEGORY}_train_split.csv'), one_hot=False, n_class=N_CLASS, starting_class=0) # starting class index 0, 17, 31 respectively
    x_dev, y_dev = load_data(os.path.join(
        data_dir, f'{BIG_CATEGORY}_valid_split.csv'), one_hot=False, n_class=N_CLASS, starting_class=0)
```



Then run for example

```
python adversarial_abblstm.py #change with any model
```



### For the models under [https://github.com/dongjun-Lee/text-classification-models-tf]:

1) Modify the csv and titles.txt path in data_utils.py (titles.txt consists of all the titles in the dataset separated by newline)

2) Modify NUM_CLASS in train.py, either 17, 14 or 27

3) Run

```
python train-py --model=word_cnn #we used word_cnn, char_cnn, word_rnn and rcnn
```

### For Fasttext:

First use the title_classification/csv_to_txt_fasttext.py to convert csv to fasttext format, change the big_category in the script.

The best params we found can be found in generate_predictions.py in the root folder



### References




