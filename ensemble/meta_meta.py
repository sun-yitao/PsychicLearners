import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
from numpy import tensordot
from numpy.linalg import norm
from scipy.optimize import differential_evolution
from pathlib import Path

psychic_learners_dir = Path.cwd().parent
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')


def ensemble_predictions(weights, big_category, subset='valid'):
	# make predictions
    if big_category == 'fashion':
        model_name = 'all_19_KNN400_50_rf_itemid'
    else:
        model_name = 'all_19_KNN200_rf_itemid'
    out_of_fold_val_1 = np.load(os.path.join(ROOT_PROBA_FOLDER, big_category, 'meta', model_name + '_nn', f'{subset}.npy'))
    out_of_fold_val_2 = np.load(os.path.join(ROOT_PROBA_FOLDER, big_category, 'meta', model_name + '_xgb', f'{subset}.npy'))
    out_of_fold_val_3 = np.load(os.path.join(ROOT_PROBA_FOLDER, big_category, 'meta', model_name + '_ada', f'{subset}.npy'))
    yhats = [out_of_fold_val_1, out_of_fold_val_2, out_of_fold_val_3]
    yhats = array(yhats)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


def evaluate_ensemble(weights, y_valid, big_category):
    # make prediction
    yhat = ensemble_predictions(weights, big_category)
    # calculate accuracy
    return accuracy_score(y_valid, yhat)

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result

# loss function for optimization process, designed to be minimized


def loss_function(weights, y_valid, big_category):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(normalized, y_valid, big_category)

def predict_all_to_csv(combined_model_name):
    beauty_weights = get_weights_for_big_category('beauty')
    beauty_predicted_test_categories = ensemble_predictions(beauty_weights, 'beauty', subset='test')
    beauty_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'beauty_test_split.csv'))
    beauty_preds = pd.DataFrame(data={'itemid': beauty_test['itemid'].values,
                                      'Category': beauty_predicted_test_categories})

    fashion_weights = get_weights_for_big_category('fashion')
    fashion_predicted_test_categories = ensemble_predictions(fashion_weights, 'fashion', subset='test')
    fashion_predicted_test_categories  = fashion_predicted_test_categories + 17
    fashion_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'fashion_test_split.csv'))
    fashion_preds = pd.DataFrame(data={'itemid': fashion_test['itemid'].values,
                                       'Category': fashion_predicted_test_categories})

    mobile_weights = get_weights_for_big_category('mobile')
    mobile_predicted_test_categories = ensemble_predictions(mobile_weights, 'mobile', subset='test')
    mobile_predicted_test_categories = mobile_predicted_test_categories + 31
    mobile_test = pd.read_csv(str(psychic_learners_dir / 'data' / 'mobile_test_split.csv'))
    mobile_preds = pd.DataFrame(data={'itemid': mobile_test['itemid'].values,
                                      'Category': mobile_predicted_test_categories})

    all_preds = pd.concat([beauty_preds, fashion_preds, mobile_preds], ignore_index=True)
    all_preds.to_csv(str(psychic_learners_dir / 'data' / 'predictions' /
                         combined_model_name) + '_weighted_metameta.csv', index=False)
    

def get_weights_for_big_category(big_category):
    VALID_CSV = str(psychic_learners_dir / 'data' / f'{big_category}_valid_split.csv')
    valid_df = pd.read_csv(VALID_CSV)
    train_y = valid_df['Category'].values
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    dummy_x = np.zeros(train_y.shape)
    X_train, X_valid, y_train, y_valid = train_test_split(dummy_x, train_y,
                                                        stratify=train_y,
                                                        test_size=0.25, random_state=42)
    # fit all models
    n_members = 3
    # evaluate averaging ensemble (equal weights)
    weights = [1.0/n_members for _ in range(n_members)]
    score = evaluate_ensemble(weights, y_valid, big_category)
    print('Equal Weights Score: %.3f' % score)
    # define bounds on each weight
    bound_w = [(0.0, 1.0) for _ in range(n_members)]
    # arguments to the loss function
    search_arg = (y_valid, big_category)
    # global optimization of ensemble weights
    result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
    # get the chosen weights
    weights = normalize(result['x'])
    print(f'Optimized {big_category} Weights: {weights}')
    # evaluate chosen weights
    score = evaluate_ensemble(weights, y_valid, big_category)
    print(f'Optimized {big_category} Weights Score: {score}')
    return weights

if __name__ == '__main__':
    predict_all_to_csv(combined_model_name='all_19_KNN400_50_rf_itemid_xgb')
