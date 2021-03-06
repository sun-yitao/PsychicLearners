import os
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import re

import plaidml.keras
plaidml.keras.install_backend()

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize, scale, minmax_scale, robust_scale, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model, ensemble
from scipy.special import softmax
from sklearn import metrics
from sklearn.externals import joblib
from nltk import word_tokenize
from tqdm import tqdm

import keras
from keras import layers
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
import tensorflow as tf
import xgboost
from catboost import CatBoostClassifier, Pool

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # workaround for macOS mkl issue

"""Stacked Ensemble using probabilties predicted on validation and test"""

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'fashion'
print(BIG_CATEGORY)
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
BATCH_SIZE = 128

# list of models to include in  stack
model_names = [
    'char_cnn',
    'extractions_fasttext',
    'image_model',
    'title_fasttext',
    'word_cnn',
    'word_rnn',
    'rcnn',
    'bert_v1',
    'nb_ngrams_2',
    'adv_abblstm',
    'atten_bilstm',
    'ind_rnn',
    'multi_head',
    'log_reg_tfidf',
    'knn_itemid_400_50',  # fashion
    #'KNN_itemid',  # non-fashion
    'knn5_tfidf',
    'knn10_tfidf',
    'knn40_tfidf',
    #'rf_itemid',  # non-fashion

]

unwanted_models = [
    'log_reg',
    'capsule_net',
    'rf',
    'rf_tfidf',
    'nb_ngrams',
    'xgb_itemid_index',
    'meta',
    'knn5',
    'knn10',
    'knn20_tfidf',
    'xgb',
    'xgb_tfidf',
    'bert_large',
    'knn80_tfidf',
    'knn160_tfidf',
    'nb_extractions',
]



N_MODELS = len(model_names)
print(f'Number Models: {N_MODELS}')
meta_model_names = []

def read_probabilties(proba_folder, subset='valid',
                      model_names=model_names):
    """Reads saved .npy validation and test predicted probabilities from PsychicLearners/data/probabilities"""
    proba_folder = Path(proba_folder)
    all_probabilities = []
    for folder in proba_folder.iterdir():
        if not folder.is_dir():
            continue
        elif model_names and folder.name not in model_names:
            if folder.name not in unwanted_models:
                print(folder.name, 'not included')
            continue
        for npy in folder.glob(f'*{subset}.npy'):
            prob = np.load(str(npy))
            if not (prob >= 0).all():
                prob = softmax(prob, axis=1)
            prob = normalize(prob, axis=1)
            #prob = scale(prob, axis=1)
            all_probabilities.append(prob)

    all_probabilities = np.concatenate([prob for prob in all_probabilities], axis=1)
    print(all_probabilities.shape)
    print(N_MODELS * N_CLASSES)
    return all_probabilities


MODEL_INPUT_SHAPE = (N_CLASSES * N_MODELS,)
def ensemble_model(dense1=None, dense2=None, n_layers=4, dropout=0.25, k_reg=0):
    """Creates NN ensemble model"""
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    if dense1:
        x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(input_tensor)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
    if dense2:
        for n in range(n_layers-1):
            x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                            kernel_regularizer=k_regularizer)(x)
            x = layers.PReLU()(x)
            if not n == n_layers-2: # Don't want dropout and BN on last layer
                x = layers.Dropout(dropout)(x)
                x = layers.BatchNormalization()(x)

    if dense1:
        predictions = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(x)
    else:
        predictions = layers.Dense(
            N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(input_tensor)

    model = keras.models.Model(inputs=input_tensor, outputs=predictions)
    return model


TWO_HEAD_SHAPE = int(N_CLASSES * N_MODELS / 2)
def two_head_ensemble_model(dense1=None, dense2=None, dropout=0.25, k_reg=0.00000001):
    """Not used due to no performance gains"""
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=(TWO_HEAD_SHAPE,))
    x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform',
                        kernel_regularizer=k_regularizer)(input_tensor)
    x = layers.PReLU()(x)
    out = layers.Dropout(dropout)(x)
    half_model = keras.models.Model(inputs=input_tensor, outputs=out)

    inp_a = keras.layers.Input(shape=(TWO_HEAD_SHAPE,))
    inp_b = keras.layers.Input(shape=(TWO_HEAD_SHAPE,))
    out_a = half_model(inp_a)
    out_b = half_model(inp_b)

    concatenated = keras.layers.concatenate([out_a, out_b])
    x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                     kernel_regularizer=k_regularizer)(concatenated)
    x= layers.PReLU()(x)
    predictions = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(x)
    model = keras.models.Model(inputs=[inp_a, inp_b], outputs=predictions)

    return model


def train_nn(dense1=150, dense2=32, n_layers=4, dropout=0.2, k_reg=0.00000001, 
             lr_base=0.01, epochs=50, lr_decay_factor=1,
             checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined_nn'),
             model_name='1', extract_probs=False):
    """Train NN ensemble, extract_probs=False will perform 4 fold CV. extract_probs=True will save probabilities against validation fold and save model"""
    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    
    kfold = StratifiedKFold(n_splits=4, random_state=7, shuffle=True)
    cvscores = []
    encoder = OneHotEncoder(sparse=False)
    if not extract_probs:
        for train, test in kfold.split(train_x, train_y):
            print(len(train))
            print(len(test))
            model = ensemble_model(dense1=dense1, dense2=dense2, n_layers=n_layers,
                                   dropout=dropout, k_reg=k_reg)
            decay = lr_base/(epochs * lr_decay_factor)
            sgd = keras.optimizers.SGD(lr=lr_base, decay=decay, momentum=0.9, nesterov=True)
            # callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_acc', min_delta=0.0001, patience=7)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5,
                                                        verbose=1, mode='auto', min_delta=0.00001,
                                                        cooldown=0, min_lr=0)
            model.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            y_train = encoder.fit_transform(train_y.reshape(-1, 1))
            model.fit(x=train_x[train], y=y_train[train], batch_size=BATCH_SIZE, epochs=1000, verbose=2, 
                    callbacks=[early_stopping, reduce_lr], validation_data=(train_x[test], y_train[test]),
                    shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)
            scores = model.evaluate(train_x[test], y_train[test], verbose=0)
            print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        print("CV ACC %.4f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    if extract_probs:
        X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
                                                              stratify=train_y,
                                                              test_size=0.25, random_state=42)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_valid = encoder.fit_transform(y_valid.reshape(-1, 1))
        test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='test')
        model = ensemble_model(dense1=dense1, dense2=dense2, n_layers=n_layers,
                               dropout=dropout, k_reg=k_reg)
        decay = lr_base/(epochs * lr_decay_factor)
        sgd = keras.optimizers.SGD(lr=lr_base, decay=decay, momentum=0.9, nesterov=True)
        # callbacks
        checkpoint_path = os.path.join(
            checkpoint_dir, '{}_checkpoints'.format(model_name))
        os.makedirs(checkpoint_path, exist_ok=True)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.00001, patience=10)
        ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.4f}.h5'),
                                               monitor='val_acc', verbose=1, save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5,
                                                        verbose=1, mode='auto', min_delta=0.00001,
                                                        cooldown=0, min_lr=0)
        model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=1000, verbose=2,
                  callbacks=[ckpt, reduce_lr, early_stopping], validation_data=(
                      X_valid, y_valid),
                  shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)
        val_preds = model.predict(X_valid)
        test_preds = model.predict(test_x)
        print(test_preds.shape)
        os.makedirs(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name + '_nn'), exist_ok=True)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name + '_nn', 'valid.npy'), val_preds)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name + '_nn', 'test.npy'), test_preds)


def train_two_head_model(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined_2_head'),
          model_name='1'):
    """Not used"""
    model = two_head_ensemble_model(dense1=200, dense2=200,  # mobile may need more dropout
                                    dropout=0.3, k_reg=0.00000001)
    decay = lr_base/(epochs * lr_decay_factor)
    sgd = keras.optimizers.SGD(lr=lr_base, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join(checkpoint_dir, '{}_checkpoints'.format(model_name))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                                  verbose=1, mode='auto',
                                                  cooldown=0, min_lr=0)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
                                                          stratify=train_y,
                                                          test_size=0.25, random_state=42)
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_valid = encoder.fit_transform(y_valid.reshape(-1, 1))
    model.fit([X_train[:, :TWO_HEAD_SHAPE], X_train[:, TWO_HEAD_SHAPE:]], y=y_train, batch_size=BATCH_SIZE, epochs=1000, verbose=2,
              callbacks=[ckpt, reduce_lr], validation_data=([X_valid[:, :TWO_HEAD_SHAPE], X_valid[:, TWO_HEAD_SHAPE:]], y_valid),
              shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)


with open("word_dict.pickle", "rb") as f:
    word_dict = pickle.load(f)


def build_word_dataset(titles, word_dict, document_max_len):
    df = pd.DataFrame(data={'title': titles})
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["title"]))
    x = list(
        map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d))
                 * [word_dict["<pad>"]], x))
    return x


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text


def batch_iter(inputs, batch_size):
    inputs = np.array(inputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(inputs))
        yield inputs[start_index:end_index]

WORD_MAX_LEN = 15
TEXT_MODEL_PATH = str(psychic_learners_dir / 'data' / 'keras_checkpoints' /
                      BIG_CATEGORY / 'word_cnn' / '0.8223667828685259.ckpt-686000')

def extract_text_features(titles, subset):
    """titles: array of titles, not used"""
    test_x = build_word_dataset(titles, word_dict, WORD_MAX_LEN)
    graph = tf.Graph()
    all_text_features = []
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                "{}.meta".format(TEXT_MODEL_PATH))
            saver.restore(sess, TEXT_MODEL_PATH)

            x = graph.get_operation_by_name("x").outputs[0]
            y = graph.get_operation_by_name("Reshape").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            batches = batch_iter(test_x, BATCH_SIZE)
            for batch_x in batches:
                feed_dict = {
                    x: batch_x,
                    is_training: False
                }

                text_features = sess.run(y, feed_dict=feed_dict)
                for text_feature in text_features:
                    all_text_features.append(text_feature)
    all_text_features = np.array(all_text_features)
    os.makedirs(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'word_cnn'), exist_ok=True)
    np.save(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'word_cnn' / f'{subset}.npy'), all_text_features)
    return all_text_features


def train_catboost(model_name, extract_probs=False, save_model=False):
    """Not used"""
    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
                                                          stratify=train_y,
                                                          test_size=0.25, random_state=42)
    classifier = CatBoostClassifier(
        iterations=150, learning_rate=0.03, depth=9, l2_leaf_reg=2,
        loss_function='MultiClass', border_count=32)
    train_data = Pool(X_train, y_train)
    valid_data = Pool(X_valid, y_valid)
    classifier.fit(train_data)
    # predict the labels on validation dataset
    predictions = classifier.predict(train_data)
    print('Train Acc: {}'.format(metrics.accuracy_score(predictions, y_train)))
    predictions = classifier.predict(valid_data)
    print('Valid accuracy: ', metrics.accuracy_score(predictions, y_valid))
    if save_model:
        checkpoint_path = psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined_catboost' / '{}_saved_model'.format(model_name)
        os.makedirs(str(checkpoint_path), exist_ok=True)
        classifier.save_model(str(checkpoint_path / 'catboost_model'))


def train_adaboost_extra_trees(model_name, extract_probs=False, save_model=False, stratified=False, param_dict=None):
    if BIG_CATEGORY == 'fashion' and 'KNN_itemid' in model_names:
        raise Exception('Warning KNN itemid in fashion')

    train_probs = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    valid_df = pd.read_csv(VALID_CSV)
    train_y = valid_df['Category'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    if param_dict:
        print(param_dict)
        classifier = xgboost.XGBClassifier(**param_dict)
    else:
        base_estim = ensemble.ExtraTreesClassifier(n_estimators=110, criterion='gini', max_depth=None, min_samples_split=2,  # 80.8766%
                                                   min_samples_leaf=1, max_features='auto')
        classifier = ensemble.AdaBoostClassifier(base_estimator=base_estim, n_estimators=60, learning_rate=1.0,
                                                 algorithm='SAMME.R')
    if stratified:
        kfold = StratifiedKFold(n_splits=4, random_state=7, shuffle=True)
        results = cross_val_score(
            classifier, train_probs, train_y, cv=kfold, n_jobs=-1, )
        print("Accuracy: %.4f%% (%.2f%%)" %
              (results.mean()*100, results.std()*100))
    elif not stratified:
        X_train, X_valid, y_train, y_valid = train_test_split(train_probs, train_y,
                                                              stratify=train_y,
                                                              test_size=0.25, random_state=42)
        classifier.fit(X_train, y_train)
        # predict the labels on validation dataset
        predictions = classifier.predict(X_train)
        print('Train Acc: {}'.format(metrics.accuracy_score(predictions, y_train)))
        predictions = classifier.predict(X_valid)
        print('Valid accuracy: ', metrics.accuracy_score(predictions, y_valid))
    if save_model:
        assert not stratified
        checkpoint_path = psychic_learners_dir / 'data' / 'keras_checkpoints' / \
            BIG_CATEGORY / 'combined_ada' / '{}_saved_model'.format(model_name)
        os.makedirs(str(checkpoint_path), exist_ok=True)
        joblib.dump(classifier, str(checkpoint_path / "adaboost.joblib"))
    if extract_probs:
        assert not stratified
        test_x = read_probabilties(proba_folder=os.path.join(
            ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='test')
        val_preds = classifier.predict_proba(X_valid)
        test_preds = classifier.predict_proba(test_x)
        print(test_preds.shape)
        os.makedirs(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY,
                                 'meta', model_name + '_ada'), exist_ok=True)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY,
                             'meta', model_name + '_ada', 'valid.npy'), val_preds)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY,
                             'meta', model_name + '_ada', 'test.npy'), test_preds)

def change_wrong_category():
    """Not used due to lack of submissions"""
    valid_df = pd.read_csv(VALID_CSV)
    with open('/Users/sunyitao/Downloads/all_corrected_wrongs.txt') as f:
        checked_itemids = f.readlines()
        checked_itemids = [itemid.replace('\n', '') for itemid in checked_itemids]
    suspected_wrong = pd.read_csv('/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/suspected_wrong_valid.csv')
    confirmed_wrong = suspected_wrong[suspected_wrong['itemid'].isin(checked_itemids)]
    categories = []
    for itemid, category in tqdm(valid_df[['itemid', 'Category']].values):
        if itemid in checked_itemids:
            category = confirmed_wrong['expected_category'][confirmed_wrong['itemid'] == itemid].values
        categories.append(category)
    valid_df['Category'] = categories #TODO this does not work
    valid_df.to_csv(str(psychic_learners_dir / 'data' / f'corrected_{BIG_CATEGORY}_valid_split.csv'))

def train_xgb(model_name, extract_probs=False, save_model=False, stratified=False, param_dict=None):
    if BIG_CATEGORY == 'fashion' and 'KNN_itemid' in model_names:
        raise Exception('Warning KNN itemid in fashion')
    """KNN itemid is gives good performance on validation but very poor performance on public leaderboard due to different itemid distributions
       extract_probs=False, save_model=False, stratified=True to perform 4 fold CV
       extract_probs=True, save_model=True, stratified=False to save out-of-fold probabilities and model"""
    train_probs = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    #train_elmo = np.load(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'elmo' / 'valid_flat.npy'))
    #train_probs = np.concatenate([train_probs, train_elmo], axis=1)
    valid_df = pd.read_csv(VALID_CSV)
    train_y = valid_df['Category'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    if param_dict:
        print(param_dict)
        classifier = xgboost.XGBClassifier(**param_dict)
    else:
        classifier = xgboost.XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=150, silent=True,
            objective='binary:logistic', booster='gbtree', n_jobs=-1, nthread=None,
            gamma=0, min_child_weight=2, max_delta_step=0, subsample=1.0, colsample_bytree=1.0,
            colsample_bylevel=1, reg_alpha=0.01, reg_lambda=1, scale_pos_weight=1,
            base_score=0.5, random_state=0, seed=None, missing=None)
    if stratified:
        kfold = StratifiedKFold(n_splits=4, random_state=7, shuffle=True)
        results = cross_val_score(classifier, train_probs, train_y, cv=kfold, n_jobs=-1, )
        print("Accuracy: %.4f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    elif not stratified:
        X_train, X_valid, y_train, y_valid = train_test_split(train_probs, train_y,
                                                              stratify=train_y,
                                                              test_size=0.25, random_state=42)
        classifier.fit(X_train, y_train)
        # predict the labels on validation dataset
        predictions = classifier.predict(X_train)
        print('Train Acc: {}'.format(metrics.accuracy_score(predictions, y_train)))
        predictions = classifier.predict(X_valid)
        print('Valid accuracy: ', metrics.accuracy_score(predictions, y_valid))
    if save_model:
        assert not stratified
        checkpoint_path = psychic_learners_dir / 'data' / 'keras_checkpoints' / \
            BIG_CATEGORY / 'combined_xgb' / '{}_saved_model'.format(model_name)
        os.makedirs(str(checkpoint_path), exist_ok=True)
        joblib.dump(classifier, str(checkpoint_path / "xgb.joblib.dat"))
    if extract_probs:
        assert not stratified
        test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='test')
        val_preds = classifier.predict_proba(X_valid)
        test_preds = classifier.predict_proba(test_x)
        print(test_preds.shape)
        os.makedirs(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name + '_xgb'), exist_ok=True)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name + '_xgb', 'valid.npy'), val_preds)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY,  'meta', model_name + '_xgb', 'test.npy'), test_preds)


def bayes_search_xgb(param_dict):
    """Bayesian optimisation of xgb parameters"""
    train_probs = read_probabilties(proba_folder=os.path.join(
        ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    valid_df = pd.read_csv(VALID_CSV)
    train_y = valid_df['Category'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)

    bayes_cv_tuner = BayesSearchCV(
        estimator=xgboost.XGBClassifier(**param_dict),
        search_spaces={
            'learning_rate': (0.01, 1.0, 'log-uniform'),
            'min_child_weight': (0, 4),
            'max_depth': (6, 9),
            'max_delta_step': (0, 20),
            'subsample': (0.7, 1.0, 'uniform'),
            'colsample_bytree': (0.7, 1.0, 'uniform'),
            'colsample_bylevel': (0.7, 1.0, 'uniform'),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'gamma': (1e-9, 0.5, 'log-uniform'),
            'n_estimators': (50, 300),
            'scale_pos_weight': (1e-6, 500, 'log-uniform')
        },
        cv=StratifiedKFold(n_splits=4, random_state=7, shuffle=True),
        scoring='accuracy',
        n_jobs=-1,
        n_iter=100,
        verbose=1,
        refit=True,
        random_state=7
    )

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest Accuracy: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))

        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name+"_cv_results.csv")
    result = bayes_cv_tuner.fit(train_probs, train_y, callback=status_print)

def predict_keras(model_path, big_category, model_names=model_names):
    test_x = read_probabilties(proba_folder=os.path.join(
        ROOT_PROBA_FOLDER, big_category), subset='test')
    model = keras.models.load_model(model_path)
    preds = model.predict(test_x)
    print(preds.shape)
    return preds


def predict_xgb(model_path, big_category, model_names=model_names):
    test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, big_category), subset='test', model_names=model_names)
    model = joblib.load(model_path)
    predictions = model.predict(test_x)
    return predictions


def predict_catboost(model_path, big_category, model_names=model_names):
    test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, big_category), subset='test')
    test_data = Pool(test_x)
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    predictions = from_file.predict(test_data)
    return predictions

def predict_all_nn():
    """For the first few versions of our ensemble we used solely NN as meta-learner. Not used in later versions"""
    beauty_preds = predict_keras(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/beauty/combined/all_13_checkpoints/model.07-0.80.h5',
        big_category='beauty')
    beauty_preds = np.argmax(beauty_preds, axis=1)
    beauty_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'beauty_test_split.csv'))
    beauty_preds = pd.DataFrame(data={'itemid':beauty_test['itemid'].values, 
                                      'Category': beauty_preds})
    
    fashion_preds = predict_keras(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/fashion/combined/all_13_checkpoints/model.12-0.68.h5',
        big_category='fashion')
    fashion_preds = np.argmax(fashion_preds, axis=1)
    fashion_preds = fashion_preds + 17
    fashion_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'fashion_test_split.csv'))
    fashion_preds = pd.DataFrame(data={'itemid': fashion_test['itemid'].values,
                                       'Category': fashion_preds})

    mobile_preds = predict_keras(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/mobile/combined/all_13_checkpoints/model.18-0.85.h5',
        big_category='mobile')
    mobile_preds = np.argmax(mobile_preds, axis=1)
    mobile_preds = mobile_preds + 31
    mobile_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'mobile_test_split.csv'))
    mobile_preds = pd.DataFrame(data={'itemid': mobile_test['itemid'].values,
                                      'Category': mobile_preds})

    all_preds = pd.concat([beauty_preds, fashion_preds, mobile_preds], ignore_index=True)
    all_preds.to_csv(str(psychic_learners_dir / 'data' / 'predictions' /
                         COMBINED_MODEL_NAME) + '.csv', index=False)


def predict_all_xgb():
    """In the second iteration of our algorithm we used xgb as meta-learner which provides better performance by about 0.5%
       pass the path to model and level-1 model names to each of the 3 predict_xgb functions"""
    beauty_preds = predict_xgb(
        f'/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/beauty/combined_xgb/all_19_KNN200_rf_itemid_saved_model/xgb.joblib.dat',
        big_category='beauty',
        model_names=[
            'char_cnn',
            'extractions_fasttext',
            'image_model',
            'title_fasttext',
            'word_cnn',
            'word_rnn',
            'rcnn',
            'bert_v1',
            'nb_ngrams_2',
            'adv_abblstm',
            'atten_bilstm',
            'ind_rnn',
            'multi_head',
            'log_reg_tfidf',
            #'KNN_itemid_200',  # fashion
            'KNN_itemid',  # non-fashion
            'knn5_tfidf',
            'knn10_tfidf',
            'knn40_tfidf',
            'rf_itemid', #non-fashion
        ]
        )
    #beauty_preds = np.argmax(beauty_preds, axis=1)
    beauty_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'beauty_test_split.csv'))
    beauty_preds = pd.DataFrame(data={'itemid': beauty_test['itemid'].values,
                                      'Category': beauty_preds})

    fashion_preds = predict_xgb(
        f'/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/fashion/combined_xgb/all_19_KNN200_rf_itemid_saved_model/xgb.joblib.dat',
        big_category='fashion', model_names=[
            'char_cnn',
            'extractions_fasttext',
            'image_model',
            'title_fasttext',
            'word_cnn',
            'word_rnn',
            'rcnn',
            'bert_v1',
            'nb_ngrams_2',
            'adv_abblstm',
            'atten_bilstm',
            'ind_rnn',
            'multi_head',
            'log_reg_tfidf',
            'KNN_itemid_200',  # fashion
            #'KNN_itemid',  # non-fashion
            'knn5_tfidf',
            'knn10_tfidf',
            'knn40_tfidf',
            #'rf_itemid',  # non-fashion
        ])
    #fashion_preds = np.argmax(fashion_preds, axis=1)
    fashion_preds = fashion_preds + 17
    fashion_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'fashion_test_split.csv'))
    fashion_preds = pd.DataFrame(data={'itemid': fashion_test['itemid'].values,
                                       'Category': fashion_preds})

    mobile_preds = predict_xgb(
        f'/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/mobile/combined_xgb/all_19_KNN200_rf_itemid_saved_model/xgb.joblib.dat',
        big_category='mobile',
        model_names=[
            'char_cnn',
            'extractions_fasttext',
            'image_model',
            'title_fasttext',
            'word_cnn',
            'word_rnn',
            'rcnn',
            'bert_v1',
            'nb_ngrams_2',
            'adv_abblstm',
            'atten_bilstm',
            'ind_rnn',
            'multi_head',
            'log_reg_tfidf',
            #'KNN_itemid_400',  # fashion
            'KNN_itemid',  # non-fashion
            'knn5_tfidf',
            'knn10_tfidf',
            'knn40_tfidf',
            'rf_itemid',  # non-fashion
        ])
    #mobile_preds = np.argmax(mobile_preds, axis=1)
    mobile_preds = mobile_preds + 31
    mobile_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'mobile_test_split.csv'))
    mobile_preds = pd.DataFrame(data={'itemid': mobile_test['itemid'].values,
                                      'Category': mobile_preds})

    all_preds = pd.concat([beauty_preds, fashion_preds,
                           mobile_preds], ignore_index=True)
    all_preds.to_csv(str(psychic_learners_dir / 'data' / 'predictions' /
                         COMBINED_MODEL_NAME) + '_xgb.csv', index=False)



def evaluate_total_accuracy(val_beauty_acc, val_fashion_acc, val_mobile_acc, kaggle_public_acc):
    """This was before we used cross validation we used a 75/25 split of validation data"""
    val_split = 0.25
    total_examples = 57317*val_split + 43941*val_split + 32065*val_split + 172402*0.3
    num_correct = 57317*val_split*val_beauty_acc + \
                  43941*val_split*val_fashion_acc + \
                  32065*val_split*val_mobile_acc + \
                  172402*0.3*kaggle_public_acc
    return num_correct/total_examples

def evaluate_cv_total_accuracy(val_beauty_acc, val_fashion_acc, val_mobile_acc, kaggle_public_acc=None):
    """Utility function to evaluate results, pass kaggle_public_acc=False to evaluate local CV score"""
    if kaggle_public_acc:
        total_examples = 57317 + 43941 + 32065 + 172402*0.3
        num_correct = 57317*val_beauty_acc + \
                    43941*val_fashion_acc + \
                    32065*val_mobile_acc + \
                    172402*0.3*kaggle_public_acc
    else:
        total_examples = 57317 + 43941 + 32065
        num_correct = 57317*val_beauty_acc + \
                    43941*val_fashion_acc + \
                    32065*val_mobile_acc

    return num_correct/total_examples


def check_output():
    """Utility function to compare a new prediction csv with one that is known to be reliable"""
    #verified_prediction_df = pd.read_csv(str(psychic_learners_dir / 'data' / 'predictions' / '17_with_itemid_xgb.csv'))
    verified_prediction_df = pd.read_csv('/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/predictions/all_19_KNN200_rf_itemid_xgb.csv')
    #unverified_prediction_df = pd.read_csv(str(psychic_learners_dir / 'data' / 'predictions' / COMBINED_MODEL_NAME) + '_xgb.csv')
    unverified_prediction_df = pd.read_csv('/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/predictions/all_19_KNN100_rf_itemid_weighted_metameta.csv')
    verified_output = verified_prediction_df['Category'].values
    unverified_output = unverified_prediction_df['Category'].values
    beauty_verified = verified_output[:76545]
    fashion_verified = verified_output[76545:131985]
    mobile_verified = verified_output[131985:]
    beauty_unverified = unverified_output[:76545]
    fashion_unverified = unverified_output[76545:131985]
    mobile_unverified = unverified_output[131985:]
    matches = np.sum(verified_output == unverified_output)
    beauty_matches = np.sum(beauty_verified == beauty_unverified)
    fashion_matches = np.sum(fashion_verified == fashion_unverified)
    mobile_matches = np.sum(mobile_verified == mobile_unverified)
    print(f'Percentage match: {matches / len(verified_output)}')
    print(f'Beauty matches: {beauty_matches / len(beauty_verified)}')
    print(f'Fashion matches: {fashion_matches / len(fashion_verified)}')
    print(f'Mobile matches: {mobile_matches / len(mobile_verified)}')

if __name__ == '__main__':
    COMBINED_MODEL_NAME = 'all_19_KNN400_50_rf_itemid'
    
    train_nn(dense1=200, dense2=48, n_layers=4, dropout=0.2, k_reg=0.00000001, lr_base=0.01, epochs=50, lr_decay_factor=1,
             checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined_nn'),
             model_name=COMBINED_MODEL_NAME, extract_probs=True)
    
    param_dict = {'min_child_weight': 3, 'gamma': 0.00043042962756640143, 'colsample_bylevel': 0.872677186090371, 'scale_pos_weight': 28.589594129413953, 'n_estimators': 137, 'n_jobs': -1,
                  'reg_alpha': 4.06423528965959e-07, 'reg_lambda': 3.621346391467108e-05, 'max_delta_step': 8, 'subsample': 0.9269871195796154, 'max_depth': 7, 'learning_rate': 0.126,
                  'colsample_bytree': 0.9963806925444163}
    train_xgb(COMBINED_MODEL_NAME, extract_probs=False, save_model=False, stratified=True, param_dict=param_dict)
    

    
    #train_catboost(COMBINED_MODEL_NAME, save_model=False)
    #print(evaluate_total_accuracy(0.79651, 0.67459, 0.8366, 0)) #5+wordrnn+rcnn
    #print(evaluate_total_accuracy(0.79860, 0.67886, 0.84633, 0.76840))  # 7+bert
    #print(evaluate_total_accuracy(0.79742, 0.67977, 0.84633, 0))  # 8+atten_bilstm
    #print(evaluate_total_accuracy(0.79798, 0.67977, 0.84633, 0))  # 8+adv
    #print(evaluate_total_accuracy(0.79881, 0.68068, 0.84658, 0.76753))  # 8+adv+attn_blstm softmax, normalize
    #print(evaluate_total_accuracy(0.80105, 0.68651, 0.85131, 0.77103))  # all_13_xgb
    #print(evaluate_total_accuracy(0.83035, 0.68651, 0.874267, 0.78882))  # 13+itemid_index
    #
    #print(evaluate_cv_total_accuracy(0.823787, 0.730594, 0.8768))
    #predict_all_xgb()
    #check_output()


"""
Logs
OVERALL
4+charcnn 0.7629462331932415
5+wordrnn 0.7628749750297908
7+bert 0.7692680688489952
8+atten_bilstm 0.7690330277591126
all_13_xgb 0.7727376190442597
13+itemid_index_nofashion 0.790656 total, 0.78882 PL
17_with_itemid fashion_KNN 400 0.79851 PL 
all_19_KNN100_rf_itemid_weighted_metameta.csv = 0.8058964 total 0.80286 PL 0.80707444 internal CV

XGB
# beauty
## 150 estimators
13 + KNN = 82.3857
13 + tfidf_logreg + KNN_itemid  = 82.4799
13 + tfidf_logreg + KNN_itemid + capsulenet = 82.3892
13 + tfidf_logreg + KNN_itemid + rf  = 82.3874
13 + tfidf_logreg + KNN_itemid + rf_tfidf = 82.4154
17_with_itemid + knn40_tfidf + rf_itemid = 82.6248

## 50 estimators
13 + tfidf_logreg + KNN_itemid = 82.2269
13 + tfidf_logreg + KNN_itemid = 82.2863 max depth 6 
13 + tfidf_logreg + KNN_itemid + knn5_tfidf = 82.3124
13 + tfidf_logreg + KNN_itemid - adv + capsulenet = 82.1920
{'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 50, 'gamma': 0, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 1.0, 
'colsample_bytree': 1.0, 'colsample_bylevel': 1, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'random_state': 0}
17_with_itemid = 82.3787% LAST PL
17_with_itemid + knn20_tfidf = 82.3369
17_with_itemid + knn40_tfidf = 82.3892
17_with_itemid + knn40_tfidf + rf_itemid = 82.5497
17_with_itemid + knn40_tfidf + rf_itemid + bert_large = 82.5497
17_with_itemid + knn40_tfidf + rf_itemid + knn80_tfidf = 82.4835

## bayesian optimised
17_with_itemid + knn40_tfidf + rf_itemid = 82.67


# fashion
# 150 estimators
13 + tfidf logreg = 68.51
13 + tfidf_logreg + KNN_itemid  = 76.3707% Does not correlate with LB, K neighbours set too low
17_with_itemid + knn40_tfidf KNNitemid 100 = 74.1017

## 50 estimators
17_with_itemid KNN 400 = 71.7963
17_with_itemid + knn40_tfidf KNNitemid 200 = 72.5655
17_with_itemid + knn40_tfidf KNNitemid 150 = 72.9684
17_with_itemid + knn40_tfidf KNNitemid 100 = 73.7968

## bayesian optimised
17_with_itemid + knn40_tfidf with KNNitemid 100 = 74.05
17_with_itemid + knn40_tfidf with KNNitemid 200 = 73.0594
17_with_itemid + knn40_tfidf with KNNitemid 50_400 = 73.8628

# mobile
## 150 estimators
17_with_itemid + knn40_tfidf + rf_itemid = 87.5974

## bayesian optimised 
17_with_itemid + knn40_tfidf + rf_itemid = 87.68

"""
"""
NN
# beauty
dense1=150, dense2=32, n_layers=4, dropout=0.0, k_reg=0.00000001 = 81.3442%
dense1=150, dense2=32, n_layers=4, dropout=0.2, k_reg=0.00000001 = 81.9706
dense1=200, dense2=32, n_layers=4, dropout=0.2, k_reg=0.00000001 = 81.9879
dense1=200, dense2=48, n_layers=4, dropout=0.2, k_reg=0.00000001 = 82.0821
dense1=200, dense2=32, n_layers=5, dropout=0.2, k_reg=0.00000001 = 81.8798
dense1=200, dense2=48, n_layers=4, dropout=0.3, k_reg=0.00000001 = 81.9862

# fashion


#mobile
dense1=200, dense2=48, n_layers=4, dropout=0.2, k_reg=0.00000001 = 86.9550

Final weights
Equal Weights Score: 0.823
Optimized beauty Weights: [2.26304628e-01 7.73323638e-01 3.71734207e-04]
Optimized beauty Weights Score: 0.8272156315422191
Equal Weights Score: 0.731
Optimized fashion Weights: [0.23452134 0.65202972 0.11344894]
Optimized fashion Weights Score: 0.7356635718186784
Equal Weights Score: 0.878
Optimized mobile Weights: [0.32213879 0.56155284 0.11630837]
Optimized mobile Weights Score: 0.8800049893975302

Equal Weights Score: 0.823
Optimized beauty Weights: [0.22207012 0.772792   0.00513788]
Optimized beauty Weights Score: 0.8272156315422191
Equal Weights Score: 0.734
Optimized fashion Weights: [0.24182756 0.75672089 0.00145155]
Optimized fashion Weights Score: 0.741398143091207
Equal Weights Score: 0.878
Optimized mobile Weights: [0.29369731 0.58890224 0.11740045]
Optimized mobile Weights Score: 0.8798802544592741
"""


#XGBoost optimised params
#Beauty
{'scale_pos_weight': 1e-06, 'max_depth': 6, 'min_child_weight': 0, 'reg_lambda': 0.0002520244082129099, 'subsample': 1.0, 'reg_alpha': 1.0, 'gamma': 1e-09, 'n_jobs': -1,
 'learning_rate': 0.04765713942024485, 'n_estimators': 300, 'colsample_bylevel': 0.7, 'max_delta_step': 0, 'colsample_bytree': 0.7}


#Fashion
{'min_child_weight': 3, 'gamma': 0.00043042962756640143, 'colsample_bylevel': 0.872677186090371, 'scale_pos_weight': 28.589594129413953, 'n_estimators': 137, 'n_jobs': -1,
 'reg_alpha': 4.06423528965959e-07, 'reg_lambda': 3.621346391467108e-05, 'max_delta_step': 8, 'subsample': 0.9269871195796154, 'max_depth': 7, 'learning_rate': 0.126,
  'colsample_bytree': 0.9963806925444163}


#Mobile
{'reg_alpha': 1.39396399160069e-05, 'max_delta_step': 14, 'min_child_weight': 1, 'n_estimators': 121, 'max_depth': 9, 'scale_pos_weight': 6.469256426969229, 
 'reg_lambda': 0.00014688520690860323, 'colsample_bylevel': 0.801039079293723, 'gamma': 0.003861473967931348, 'subsample': 0.72444526943941, 'n_jobs': -1,
'colsample_bytree': 0.9291329385418159, 'learning_rate': 0.08916516649776571}
