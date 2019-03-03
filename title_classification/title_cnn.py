import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
LR_BASE = 100.0
EPOCHS = 500
data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))
valid = pd.read_csv(os.path.join(data_directory, 'valid_split.csv'))
train_x, train_y = train['title'].values, train['Category'].values
valid_x, valid_y = valid['title'].values, valid['Category'].values
y_train = keras.utils.np_utils.to_categorical(train_y)
y_valid = keras.utils.np_utils.to_categorical(valid_y)


def ConvolutionalBlock(input_shape, num_filters):
    model = keras.models.Sequential()
    #1st conv layer
    model.add(Conv1D(filters=num_filters, kernel_size=3,
                     strides=1, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #2nd conv layer
    model.add(Conv1D(filters=num_filters,
                     kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    return model

def top_kmax(x):
    x = tf.transpose(x, [0, 2, 1])
    k_max = tf.nn.top_k(x, k=top_k)
    return tf.reshape(k_max[0], (-1, num_filters[-1]*top_k))

def get_char_dict():
    char_dict = {}
    for i, c in enumerate(ALPHABET):
        char_dict[c] = i+1
    return char_dict

def char2vec(text, max_length=FEATURE_LEN):
    char_dict = get_char_dict()
    data = np.zeros(max_length)
    for i in range(0, len(text)):
        if i >= max_length:
            return data
        elif text[i] in char_dict:
            data[i] = char_dict[text[i]]
        else:
            data[i] = 68
    return data

def conv_shape(conv):
    return conv.get_shape().as_list()[1:]

replace_ip = re.compile(r'([0-9]+)(?:\.[0-9]+){3}',)

def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    #Replace IP address
    text = replace_ip.sub('', text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def vdcnn_model(num_filters, num_classes, sequence_max_length, num_chars, embedding_size, top_k, learning_rate=0.001):
    inputs = Input(shape=(sequence_max_length, ), dtype='int32', name='input')

    embedded_seq = Embedding(num_chars, embedding_size,
                                input_length=sequence_max_length)(inputs)
    embedded_seq = BatchNormalization()(embedded_seq)
    #1st Layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2,
                    padding="same")(embedded_seq)

    #ConvBlocks
    for i in range(len(num_filters)):
        conv = ConvolutionalBlock(conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    #fully connected layers
    # in original paper they didn't used dropouts
    fc1 = Dense(512, activation='relu', kernel_initializer='he_normal')(k_max)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(512, activation='relu', kernel_initializer='he_normal')(fc1)
    fc2 = Dropout(0.3)(fc2)
    out = Dense(num_classes, activation='sigmoid')(fc2)

    #optimizer
    #sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)

    model = keras.models.Model(inputs=inputs, outputs=out)
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy', metrics=['accuracy'])
    return model

num_filters = [64, 128, 256, 512]
model = vdcnn_model(num_filters=num_filters, num_classes=6, num_chars=69,
                    sequence_max_length=FEATURE_LEN, embedding_size=16, top_k=3)
model.summary()
"""
def create_embedding_matrix(filepath, word_index, embedding_dim):
    # Adding again 1 because of reserved 0 index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_x)
X_train = tokenizer.texts_to_sequences(train_x)
X_valid = tokenizer.texts_to_sequences(valid_x)
vocab_size = len(tokenizer.word_index) + 1
maxlen = 10
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_valid = pad_sequences(X_valid, padding='post', maxlen=maxlen)

embedding_dim = 100

embedding_matrix = create_embedding_matrix(
            'data/glove_word_embeddings/glove.6B.50d.txt',
            tokenizer.word_index, embedding_dim)
model = keras.models.Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model = keras.models.Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(58, activation='softmax'))
decay = LR_BASE/(EPOCHS)
sgd = keras.optimizers.SGD(lr=LR_BASE, decay=decay,
                           momentum=0.9, nesterov=True)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print(X_train.shape)
print(y_train.shape)
history = model.fit(X_train, y_train,
                    epochs=1000,
                    verbose=True,
                    validation_data=(X_valid, y_valid),
                    batch_size=4096)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_valid, y_valid, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))"""
