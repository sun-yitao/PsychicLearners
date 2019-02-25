import keras
from imblearn.keras import balanced_batch_generator
from sklearn.utils import class_weight
#X is path to image, y is category
training_generator, steps_per_epoch = balanced_batch_generator(X, y, sampler=RandomUnderSampler(), batch_size=64, random_state=42)
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_train),
                                                  y_train)
class_weight_dict = dict(enumerate(class_weight))
#possible to undersample by having multiple splits and stop every 5 epochs and restart again using a different split
