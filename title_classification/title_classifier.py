#%%
import os
import string
import pandas as pd
import xgboost
import numpy as np
import textblob #may need to run 'import nltk' and 'nltk.download('averaged_perceptron_tagger')'

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import keras
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

RANDOM_STATE = 42

# load dataset
data_directory = os.path.join('..', 'data')
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'valid_split.csv'))
train_x, train_y = train['title'], train['Category']
valid_x, valid_y = valid['title'], valid['Category']
samplers = [
    ['Random_Undersample', RandomUnderSampler(random_state=RANDOM_STATE)],
    ['ADASYN', ADASYN(random_state=RANDOM_STATE)],
    ['ROS', RandomOverSampler(random_state=RANDOM_STATE)],
    ['SMOTE', SMOTE(random_state=RANDOM_STATE)],
]


class DummySampler(object):
    """Does not perform any sampling"""
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return self.sample(X, y)

sampler = DummySampler() #Change this with any sampler if desired
print(train_x.head(10))
# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', strip_accents='unicode', #Stop words may not be needed as they seem to be already removed
                             stop_words='english', token_pattern=r'\w{1,}')
count_vect.fit(train['title'])

# transform the training and validation data using count vectorizer object

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', strip_accents='unicode',
                             stop_words='english', token_pattern=r'\w{1,}')
tfidf_vect.fit(train['title'])

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', strip_accents='unicode',
                                   stop_words='english', token_pattern=r'\w{1,}',
                                   ngram_range=(1, 3))
tfidf_vect_ngram.fit(train['title'])

"""
# load the pre-trained word-embedding vectors
print('Loading word2vec')
embeddings_index = {}
for i, line in enumerate(open('crawl-300d-2M.vec')): #This for loop takes FOREVER to run, need to find a better way to load word2vec
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(train['title'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(
    token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(
    token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector"""

#train['char_count'] = train['title'].apply(len)
#train['word_count'] = train['title'].apply(lambda x: len(x.split()))
#train['word_density'] = train['char_count'] / (train['word_count']+1)
"""
pos_family = {
    'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
    'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
    'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'adj':  ['JJ', 'JJR', 'JJS'],
    'adv': ['RB', 'RBR', 'RBS', 'WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

train['noun_count'] = train['title'].apply(lambda x: check_pos_tag(x, 'noun'))
train['verb_count'] = train['title'].apply(lambda x: check_pos_tag(x, 'verb'))
train['adj_count'] = train['title'].apply(lambda x: check_pos_tag(x, 'adj'))
train['adv_count'] = train['title'].apply(lambda x: check_pos_tag(x, 'adv'))
train['pron_count'] = train['title'].apply(lambda x: check_pos_tag(x, 'pron'))

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(
    n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(
        topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words)) """

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    if isinstance(classifier, xgboost.XGBClassifier):
        feature_vector_train = feature_vector_train.to_csc()
        feature_vector_valid = feature_vector_valid.to_csc()
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

"""
# Commented out as already tested
# Naive Bayes on Count Vectors
accuracy = train_model(make_pipeline_imb(count_vect, sampler, naive_bayes.MultinomialNB()),
                       train_x, train_y, valid_x)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect, sampler, naive_bayes.MultinomialNB()),
                       train_x, train_y, valid_x)
print("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect_ngram, sampler, naive_bayes.MultinomialNB()),
                       train_x, train_y, valid_x)
print("NB, N-Gram Vectors: ", accuracy)

# Linear Classifier on Count Vectors
accuracy = train_model(make_pipeline_imb(count_vect, sampler, 
                                         linear_model.LogisticRegression(solver='sag', n_jobs=6, multi_class='multinomial',
                                                                         tol=1e-4, C=1.e4 / 533292)),
                       train_x, train_y, valid_x)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect_ngram, sampler, 
                                         linear_model.LogisticRegression(solver='sag', n_jobs=6, multi_class='multinomial',
                                                                         tol=1e-4, C=1.e4 / 533292)),
                       train_x, train_y, valid_x)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect_ngram, sampler, 
                                         linear_model.LogisticRegression(solver='sag', n_jobs=6, multi_class='multinomial',
                                                                         tol=1e-4, C=1.e4 / 533292)),
                       train_x, train_y, valid_x)
print("LR, N-Gram Vectors: ", accuracy)"""

#accuracy = train_model(make_pipeline_imb(tfidf_vect_ngram, sampler, svm.SVC(tol=.1, max_iter=5000, verbose=True)), 
#                        train_x, train_y, valid_x)
#print("SVM, N-Gram Vectors: ", accuracy)

# RF on Count Vectors
accuracy = train_model(make_pipeline_imb(count_vect, sampler, ensemble.RandomForestClassifier(n_estimators=10, max_depth=58*10, min_samples_leaf=10)),
                       train_x, train_y, valid_x)
print("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect, sampler, ensemble.RandomForestClassifier(n_estimators=10, max_depth=58*10, min_samples_leaf=10)),
                       train_x, train_y, valid_x)
print("RF, WordLevel TF-IDF: ", accuracy)

# Extereme Gradient Boosting on Count Vectors
accuracy = train_model(make_pipeline_imb(count_vect, sampler, xgboost.XGBClassifier()),
                       train_x, train_y, valid_x)
print("Xgb, Count Vectors: ", accuracy)

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(make_pipeline_imb(tfidf_vect, sampler, xgboost.XGBClassifier()),
                       train_x, train_y, valid_x)
print("Xgb, WordLevel TF-IDF: ", accuracy)

# Try out imblearn classifiers
accuracy = train_model(make_pipeline(count_vect, BalancedRandomForestClassifier()),
                       train_x, train_y, valid_x)
print("BRF, Count Vectors: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram, BalancedRandomForestClassifier()),
                       train_x, train_y, valid_x)
print("BRF, WordLevel TF-IDF: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram,BalancedRandomForestClassifier()),
                       train_x, train_y, valid_x)
print("BRF, N-Gram Vectors: ", accuracy)

accuracy = train_model(make_pipeline(count_vect, EasyEnsembleClassifier(n_jobs=6)),
                       train_x, train_y, valid_x)
print("EEC, Count Vectors: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram, EasyEnsembleClassifier(n_jobs=6)),
                       train_x, train_y, valid_x)
print("EEC, WordLevel TF-IDF: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram, EasyEnsembleClassifier(n_jobs=6)),
                       train_x, train_y, valid_x)
print("EEC, N-Gram Vectors: ", accuracy)

accuracy = train_model(make_pipeline(count_vect, RUSBoostClassifier()),
                       train_x, train_y, valid_x)
print("RUSB, Count Vectors: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram, RUSBoostClassifier()),
                       train_x, train_y, valid_x)
print("RUSB, WordLevel TF-IDF: ", accuracy)
accuracy = train_model(make_pipeline(tfidf_vect_ngram, RUSBoostClassifier()),
                       train_x, train_y, valid_x)
print("RUSB, N-Gram Vectors: ", accuracy)

"""
def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier


classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram,
                       train_y, valid_x, is_neural_net=True)


def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(
        word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(
        100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y,
                       valid_seq_x, is_neural_net=True)
print "CNN, Word Embeddings",  accuracy


def create_rnn_lstm():
     # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(
        word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y,
                       valid_seq_x, is_neural_net=True)
print "RNN-LSTM, Word Embeddings",  accuracy


def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(
        word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y,
                       valid_seq_x, is_neural_net=True)
print "RNN-GRU, Word Embeddings",  accuracy

def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(
        word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y,
                       valid_seq_x, is_neural_net=True)
print "RNN-Bidirectional, Word Embeddings",  accuracy


def create_rcnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(
        word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(
        50, return_sequences=True))(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(
        100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rcnn()
accuracy = train_model(classifier, train_seq_x, train_y,
                       valid_seq_x, is_neural_net=True)
print "CNN, Word Embeddings",  accuracy
"""
