import os
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import re

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize, scale, minmax_scale, robust_scale, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from scipy.special import softmax
from sklearn import metrics
from sklearn.externals import joblib
from nltk import word_tokenize
from tqdm import tqdm

import xgboost
#from catboost import CatBoostClassifier, Pool

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # workaround for macOS mkl issue
"""Stacking Ensemble using probabilties predicted on validation, validating on public test set
    probs from ml-ensemble, fasttext, bert, combined-features classifier"""

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'beauty'
print(BIG_CATEGORY)
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' / '{}_train_split.csv'.format(BIG_CATEGORY))
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' / '{}_valid_split.csv'.format(BIG_CATEGORY))
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' / '{}_test_split.csv'.format(BIG_CATEGORY))
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
BATCH_SIZE = 64
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
    #'KNN_itemid_400',
    'KNN_itemid',
    'knn5_tfidf',
    'knn10_tfidf',
    #'blend_ensemble',
    #'knn20_tfidf',
    'knn40_tfidf',
    #'rf_itemid',
    #'xgb',
    #'xgb_tfidf',
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
]

if BIG_CATEGORY == 'fashion' and 'KNN_itemid' in model_names:
    raise Exception('Warning KNN itemid in fashion')

N_MODELS = len(model_names)
print('Number Models: {}'.format(N_MODELS))
meta_model_names = []

def read_probabilties(proba_folder, subset='valid',
                      model_names=model_names):
    proba_folder = Path(proba_folder)
    all_probabilities = []
    for folder in proba_folder.iterdir():
        if not folder.is_dir():
            continue
        elif model_names and folder.name not in model_names:
            if folder.name not in unwanted_models:
                print(folder.name, 'not included')
            continue
        for npy in folder.glob('*{}.npy'.format(subset)):
            prob = np.load(str(npy))
            if not (prob >= 0).all():
                prob = softmax(prob, axis=1)
            prob = normalize(prob, axis=1)
            #prob = scale(prob, axis=1)
            all_probabilities.append(prob)

    all_probabilities = np.concatenate([prob for prob in all_probabilities], axis=1)
    print(all_probabilities.shape)
    #all_probabilities = minmax_scale(all_probabilities, axis=1)
    return all_probabilities


MODEL_INPUT_SHAPE = (N_CLASSES * N_MODELS +840,)
def ensemble_model(dense1=None, dense2=None, dropout=0.25, k_reg=0):
    k_regularizer = keras.regularizers.l2(k_reg)
    input_tensor = keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    if dense1:
        x = layers.Dense(dense1, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(input_tensor)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
    if dense2:
        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(dense2, activation=None, kernel_initializer='he_uniform',
                         kernel_regularizer=k_regularizer)(x)
        x = layers.PReLU()(x)

    if dense1:
        predictions = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(x)
    else:
        predictions = layers.Dense(
            N_CLASSES, activation='softmax', kernel_regularizer=k_regularizer)(input_tensor)

    model = keras.models.Model(inputs=input_tensor, outputs=predictions)
    return model


TWO_HEAD_SHAPE = int(N_CLASSES * N_MODELS / 2)
def two_head_ensemble_model(dense1=None, dense2=None, dropout=0.25, k_reg=0.00000001):
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

def train_nn(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined'),
          model_name='1', extract_probs=False):
    model = ensemble_model(dense1=100, dense2=32,   #mobile may need more dropout
                           dropout=0.2, k_reg=0.00000001)
    decay = lr_base/(epochs * lr_decay_factor)
    sgd = keras.optimizers.SGD(
        lr=lr_base, decay=decay, momentum=0.9, nesterov=True)

    # callbacks
    checkpoint_path = os.path.join( checkpoint_dir, '{}_checkpoints'.format(model_name))
    os.makedirs(checkpoint_path, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                           monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    
    kfold = StratifiedKFold(n_splits=4, random_state=7, shuffle=True)
    cvscores = []
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(train_y.reshape(-1, 1))
    for train, test in kfold.split(train_x, y_train):
        model.fit(x=train_x[train], y=y_train[train], batch_size=BATCH_SIZE, epochs=1000, verbose=2, 
                  callbacks=[ckpt, early_stopping], validation_data=(train_x[test], y_train[test]),
                  shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)
        scores = model.evaluate(train_x[test], y_train[test], verbose=0)
        print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.4f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    if extract_probs:
        X_train, X_valid, y_train, y_valid = train_test_split(train_x, y_train,
                                                              stratify=y_train,
                                                              test_size=0.25, random_state=42)
        test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='test')
        model = ensemble_model(dense1=100, dense2=32, dropout=0.2, k_reg=0.00000001)
        decay = lr_base/(epochs * lr_decay_factor)
        sgd = keras.optimizers.SGD(lr=lr_base, decay=decay, momentum=0.9, nesterov=True)
        # callbacks
        checkpoint_path = os.path.join(
            checkpoint_dir, '{}_checkpoints'.format(model_name))
        os.makedirs(checkpoint_path, exist_ok=True)
        ckpt = keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, 'model.{epoch:02d}-{val_acc:.2f}.h5'),
                                               monitor='val_acc', verbose=1, save_best_only=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5)

        model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=1000, verbose=2, #TODO
                  callbacks=[ckpt, early_stopping], validation_data=(X_valid, y_valid),
                  shuffle=True, class_weight=None, steps_per_epoch=None, validation_steps=None)
        val_preds = model.predict(X_valid)
        test_preds = model.predict(test_x)
        print(test_preds.shape)
        os.makedirs(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name), exist_ok=True)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name, 'valid.npy'), val_preds)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name, 'test.npy'), test_preds)


def train_two_head_model(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined_2_head'),
          model_name='1'):
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
    """titles: array of titles"""
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
    np.save(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'word_cnn' / '{}.npy'.format(subset)), all_text_features)
    return all_text_features


def train_catboost(model_name, extract_probs=False, save_model=False):
    train_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    train_y = pd.read_csv(VALID_CSV)['Category'].values
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y,
                                                          stratify=train_y,
                                                          test_size=0.25, random_state=42)
    #encoder = OneHotEncoder(sparse=False)
    #y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    #y_valid = encoder.fit_transform(y_valid.reshape(-1, 1))
    """
    classifier = CatBoostClassifier(
        iterations=150, learning_rate=0.03, depth=8, l2_leaf_reg=3, 
        loss_function='MultiClass', border_count=32)"""
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

def change_wrong_category():
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
    valid_df.to_csv(str(psychic_learners_dir / 'data' / 'corrected_{}_valid_split.csv'))

def train_xgb(model_name, extract_probs=False, save_model=False, stratified=False, param_dict=None):
    train_probs = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='valid')
    #train_elmo = np.load(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'elmo' / 'valid_flat.npy'))
    #train_probs = np.concatenate([train_probs, train_elmo], axis=1)
    valid_df = pd.read_csv(VALID_CSV)
    train_y = valid_df['Category'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    """
    train_text_features = np.load(str(psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'word_cnn' / 'valid.npy'))
    #train_x = np.concatenate([train_probs, train_text_features], axis=1)
    count_vect = CountVectorizer(analyzer='word', strip_accents='unicode', 
                                 stop_words=None, ngram_range=(1, 3))
    title_encoded = count_vect.fit_transform(valid_df['title'])
    title_encoded = title_encoded.toarray()
    print(title_encoded[0])
    print(title_encoded.shape, train_probs.shape, train_text_features.shape)
    train_x = np.concatenate([train_probs, title_encoded], axis=1)"""
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
    elif not stratified and not save_model:
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
        classifier.fit(train_probs, train_y)
        checkpoint_path = psychic_learners_dir / 'data' / 'keras_checkpoints' / \
            BIG_CATEGORY / 'combined_xgb' / '{}_saved_model'.format(model_name)
        os.makedirs(str(checkpoint_path), exist_ok=True)
        joblib.dump(classifier, str(checkpoint_path / "xgb.joblib.dat"))
    if extract_probs:
        test_x = read_probabilties(proba_folder=os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY), subset='test')
        #val_preds = classifier.predict_proba(X_valid)
        test_preds = classifier.predict_proba(test_x)
        print(test_preds.shape)
        os.makedirs(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name), exist_ok=True)
        #np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name, 'valid.npy'), val_preds)
        np.save(os.path.join(ROOT_PROBA_FOLDER, BIG_CATEGORY, 'meta', model_name, 'test.npy'), test_preds)

def meta_meta_learner():
    meta_xgb_val_prob = np.load(os.path.join(ROOT_PROBA_FOLDER, model_name, 'valid.npy'))
    meta_xgb_test_prob = np.load(os.path.join(ROOT_PROBA_FOLDER, model_name, 'test.npy'))
    meta_nn_prob = np.load(os.path.join(ROOT_PROBA_FOLDER, model_name, 'valid.npy'))
    meta_nn_prob = np.load(os.path.join(ROOT_PROBA_FOLDER, model_name, 'test.npy'))
    #meta_meta_classifier = sklearn.LinearRegression()

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
    beauty_preds = predict_xgb(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/beauty/combined_xgb/13+itemid_saved_model/xgb.joblib.dat',
        big_category='beauty')
    #beauty_preds = np.argmax(beauty_preds, axis=1)
    beauty_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'beauty_test_split.csv'))
    beauty_preds = pd.DataFrame(data={'itemid': beauty_test['itemid'].values,
                                      'Category': beauty_preds})

    fashion_preds = predict_xgb(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/fashion/combined_xgb/13+itemid_saved_model/xgb.joblib.dat',
        big_category='fashion')
    #fashion_preds = np.argmax(fashion_preds, axis=1)
    fashion_preds = fashion_preds + 17
    fashion_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'fashion_test_split.csv'))
    fashion_preds = pd.DataFrame(data={'itemid': fashion_test['itemid'].values,
                                       'Category': fashion_preds})

    mobile_preds = predict_xgb(
        '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/keras_checkpoints/mobile/combined_xgb/13+itemid_saved_model/xgb.joblib.dat',
        big_category='mobile')
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
    val_split = 0.25
    total_examples = 57317*val_split + 43941*val_split + 32065*val_split + 172402*0.3
    num_correct = 57317*val_split*val_beauty_acc + \
                  43941*val_split*val_fashion_acc + \
                  32065*val_split*val_mobile_acc + \
                  172402*0.3*kaggle_public_acc
    return num_correct/total_examples

def check_output():
    verified_prediction_df = pd.read_csv(str(
        psychic_learners_dir / 'data' / 'predictions' / 'all_13_xgb.csv'))
    unverified_prediction_df = pd.read_csv(str(psychic_learners_dir / 'data' / 'predictions' /
                                               COMBINED_MODEL_NAME) + '_xgb.csv')
    verified_output = verified_prediction_df['Category'].values
    unverified_output = unverified_prediction_df['Category'].values
    matches = np.sum(verified_output == unverified_output)
    print('Percentage match: {matches / len(verified_output)}')


if __name__ == '__main__':
    COMBINED_MODEL_NAME = '17_with_itemid'
    """
    train_nn(lr_base=0.01, epochs=50, lr_decay_factor=1,
          checkpoint_dir=str(psychic_learners_dir / 'data' / 'keras_checkpoints' / BIG_CATEGORY / 'combined'),
          model_name=COMBINED_MODEL_NAME)"""
    #change_wrong_category()
    #predict_all_nn()
    #check_output()
    #train_xgb(COMBINED_MODEL_NAME, extract_probs=True, save_model=True, stratified=False)
    
    param_dict = {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 50, 'gamma': 0, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 1.0, 'n_jobs': -1, 'verbosity':2,
                  'colsample_bytree': 1.0, 'colsample_bylevel': 1, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'random_state': 0, 'tree_method':'gpu_hist'}
    train_xgb(COMBINED_MODEL_NAME, extract_probs=False,
              save_model=False, stratified=True, param_dict=param_dict)
    """
    param_dict = {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 50,
                  'gamma': 0, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 1.0, 'colsample_bytree': 1.0,
                  'colsample_bylevel': 1, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1,
                  'base_score': 0.5, 'random_state': 0}
    train_xgb(COMBINED_MODEL_NAME, extract_probs=False, save_model=False, stratified=True, param_dict=param_dict)

    param_dict = {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 50,
                  'gamma': 0, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 0.9, 'colsample_bytree': 1.0,
                  'colsample_bylevel': 1, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1,
                  'base_score': 0.5, 'random_state': 0}
    train_xgb(COMBINED_MODEL_NAME, extract_probs=False, save_model=False, stratified=True, param_dict=param_dict)

    param_dict = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 50,
                  'gamma': 0, 'min_child_weight': 2, 'max_delta_step': 0, 'subsample': 0.9, 'colsample_bytree': 1.0,
                  'colsample_bylevel': 1, 'reg_alpha': 0.01, 'reg_lambda': 1, 'scale_pos_weight': 1,
                  'base_score': 0.5, 'random_state': 0}
    train_xgb(COMBINED_MODEL_NAME, extract_probs=False,
              save_model=False, stratified=True, param_dict=param_dict)"""

    
    #train_catboost(COMBINED_MODEL_NAME, save_model=False)
    #print(evaluate_total_accuracy(0.79651, 0.67459, 0.8366, 0)) #5+wordrnn+rcnn
    #print(evaluate_total_accuracy(0.79860, 0.67886, 0.84633, 0.76840))  # 7+bert
    #print(evaluate_total_accuracy(0.79742, 0.67977, 0.84633, 0))  # 8+atten_bilstm
    #print(evaluate_total_accuracy(0.79798, 0.67977, 0.84633, 0))  # 8+adv
    #print(evaluate_total_accuracy(0.79881, 0.68068, 0.84658, 0.76753))  # 8+adv+attn_blstm softmax, normalize
    #print(evaluate_total_accuracy(0.80105, 0.68651, 0.85131, 0.77103))  # all_13_xgb
    #print(evaluate_total_accuracy(0.83035, 0.68651, 0.874267, 0.78882))  # 13+itemid_index
    

    #predict_all_xgb()
    #check_output()
