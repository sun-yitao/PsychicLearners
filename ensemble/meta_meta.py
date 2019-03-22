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
BIG_CATEGORY = 'fashion'
ROOT_PROBA_FOLDER = str(psychic_learners_dir / 'data' / 'probabilities')
TRAIN_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / f'{BIG_CATEGORY}_test_split.csv')
N_CLASSES_FOR_CATEGORIES = {'beauty': 17, 'fashion': 14, 'mobile': 27}
N_CLASSES = N_CLASSES_FOR_CATEGORIES[BIG_CATEGORY]
BATCH_SIZE = 64

def ensemble_predictions(weights):
	# make predictions
    out_of_fold_val_1 = np.load()
    out_of_fold_val_2 = np.load()
    out_of_fold_val_3 = np.load()
    yhats = [out_of_fold_val_1, out_of_fold_val_2, out_of_fold_val_3]
    yhats = array(yhats)
    # weighted sum across ensemble members
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


def evaluate_ensemble(weights, test_y):
    # make prediction
    yhat = ensemble_predictions(weights)
    # calculate accuracy
    return accuracy_score(test_y, yhat)

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
def loss_function(weights, test_y):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return 1.0 - evaluate_ensemble(normalized, test_y)


valid_df = pd.read_csv(VALID_CSV)
train_y = valid_df['Category'].values
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
dummy_x = np.zeros(train_y.shape)
X_train, X_valid, y_train, y_valid = train_test_split(dummy_x, train_y,
                                                      stratify=train_y,
                                                      test_size=0.25, random_state=42)
test_y = y_train
# fit all models
n_members = 5
# evaluate averaging ensemble (equal weights)
weights = [1.0/n_members for _ in range(n_members)]
score = evaluate_ensemble(weights, test_y)
print('Equal Weights Score: %.3f' % score)
# define bounds on each weight
bound_w = [(0.0, 1.0) for _ in range(n_members)]
# arguments to the loss function
search_arg = (test_y)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score = evaluate_ensemble(weights, test_y)
print('Optimized Weights Score: %.3f' % score)
