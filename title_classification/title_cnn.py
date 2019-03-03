import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
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
"""
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
              metrics=['accuracy'])"""
model = keras.models.Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1000, activation='relu'))
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
print("Testing Accuracy:  {:.4f}".format(accuracy))
