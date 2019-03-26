# PsychicLearners

## Overview

Our algorithm consists of 19 of level 1 algorithms and 3 level 2 algorithms trained on the predicted probabilities of the level 1 algorithms. Finally we do a weighted average of the level 2 output probabilities with the weights optimised by differential evolution.  Due to a lack of time, we were unable to do cross-validation on many of the more computationally expensive algorithms and a proper parameter search on the best configuration of the level 1 algorithms that leads to the best accuracy of the ensemble as a whole. We also apologise for the poor code quality as it was written in a hurry. A more detailed writeup can be found here: https://www.kaggle.com/c/ndsc-beginner/discussion/85396

#### Level 1 algorithms:

From https://github.com/TobiasLee/Text-Classification

1. Adversarial Training Methods for Semi-Supervised Text Classification [https://arxiv.org/abs/1605.07725]
2. Attention-Based Bidirectional Long Short-Term Memory Networks [http://www.aclweb.org/anthology/P16-2034]
3. Independently Recurrent Neural Network [https://arxiv.org/abs/1803.04831]
4. Multihead Attention Module [https://arxiv.org/abs/1706.03762]



From https://github.com/dongjun-Lee/text-classification-models-tf

5. Word-level CNN [https://arxiv.org/abs/1408.5882]
6. Character-level CNN [https://arxiv.org/abs/1509.01626]
7. Word-level Bi-RNN [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761987.pdf]
8. RCNN [https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552]

From sci-kit learn library:

9. K-nearest neighbours classification on item id with k set as 21 and 4 for beauty and mobile respectively. For fashion we found that the distributions of item IDs within 750000000 and 1500000000 does not correlate as much between the test and train sets so we used k=400 for values within this range and k=50 for values outside this range.
10. Random Forest on item IDs for beauty and mobile. We did not use this for fashion as we   were running out of submissions and did not want to risk wasting a submission for a 0.1% improvement.
11. K-nearest neighbours with K set as  5, 10, 40 on TFIDF word vectors
12. Logistic Regression on TFIDF word vectors with n-gram range 1 to 10
13. Multinomial Naive Bayes on count word vectors with n-gram range 1 to 10
   
	
Others:

16. Fasttext on titles [https://arxiv.org/abs/1607.01759], this is one of our strongest individual performers, netting us with 75% accuracy on its own.
17. Fasttext on extracted text from images with pytesseract [https://github.com/madmaze/pytesseract]
18. BERT base multi-language uncased fine-tuning on titles [https://github.com/google-research/bert]
19. Image classification: We use a SEResNext50 [https://arxiv.org/abs/1709.01507] for fashion images and a pre-trained Inception Resnet V2 [https://arxiv.org/abs/1602.07261] for mobile and beauty images, with random erasing augmentation [https://arxiv.org/abs/1708.04896v2]. We did not have time to test out different models due to the large quantity of data so we just went with what seemed to perform well in the first few epochs. We also resized images to 240x240 to speed up training and used class weights to balance the class imbalance. Training was conducted on a P6000 and 2 Tesla T4 GPUs. Image classification had a roughly 10% lower accuracy compared to text classification and we were not able to improve it by much due to time constraints.

## Steps to reproduce meta model results:

Download models and predicted probabilties from google drive

[Fasttext models](https://drive.google.com/drive/folders/1ZT3ptVDoqgHGDcWPe-3FA0L64UhJqosh?usp=sharing) 
[Our CSVs with extracted text from images](https://drive.google.com/drive/folders/1ZT3ptVDoqgHGDcWPe-3FA0L64UhJqosh?usp=sharing)
[checkpoints for our models](https://drive.google.com/open?id=1IDXhF4YwbDK99a5LRkHWzx0ZngPkab96)
[Saved probabilities in .npy format](https://drive.google.com/open?id=1gPG6_6qL5fRxO_s4I0rv111vP3wRoHBY)

Put them in PsychicLearners/data/probabilities and PsychicLearners/data/keras_checkpoints (keras checkpoints is misnamed, it actually just refers to all checkpoints keras or non-keras)

For training use the train_nn, train_xgb and train_adaboost_extra_trees functions. To run the final layer run meta_meta.py

## Steps to reproduce level 1 models

### First prepare dataset

1. Put all the original csvs and images into PsychicLearners/data and PsychicLearners/data/image/original

2. Modify data_utils/prepare_dataset.py.  The script will do a 80/20 stratified split of train.csv and the dataframes will be saved in variables train, valid and test. We didn't use most of the functions in here as they did not seem to help improve performance.

3. To get text extractions from image run get_text_extractions_parallel(dataframe)

4. make_csvs() will split train, valid and test dataframe into the 3 big categories (beauty, fashion and mobile) and persist to disk

5. copy_images_to_directory() moves the images into folders 0-57 such that in can be read by keras image generator flow_from_directory. We originally intended to use this format to make it easier to do class balancing and offline augmentation, but in the end we did train-time augmentation and used class weights to avoid increasing the size of the already large dataset. This function was badly written as we did originally did not know the 3 big categories given can be used for classification. 

For the rest of the cleaning and translating functions, we found that they did not help improve performance and did not use them in the end.



### For the models under https://github.com/TobiasLee/Text-Classification :

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



### For the models under https://github.com/dongjun-Lee/text-classification-models-tf :

1) Modify the csv and titles.txt path in data_utils.py (titles.txt consists of all the titles in the dataset separated by newline)

2) Modify NUM_CLASS in train.py, either 17, 14 or 27

3) Run

```
python train-py --model=word_cnn #we used word_cnn, char_cnn, word_rnn and rcnn
```

### For Fasttext:

First use the title_classification/csv_to_txt_fasttext.py to convert csv to fasttext format, change the big_category in the script.



