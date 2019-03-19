from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from fastText import train_supervised, train_unsupervised, load_model
import numpy as np


def print_results(N, p, r):
    print("N\t" + str(N))
    print("Accuracy: {}".format(correct/N))
    print("P@{}\t{:.5f}".format(1, p))
    print("R@{}\t{:.5f}".format(1, r))

BIG_CATEGORY = 'fashion'
if __name__ == "__main__":
    psychic_learners_dir = os.path.split(os.getcwd())[0]
    train_data = os.path.join(psychic_learners_dir, 'data', f'{BIG_CATEGORY}_train_split.txt')
    valid_data = os.path.join(psychic_learners_dir, 'data', f'{BIG_CATEGORY}_valid_split.txt')
    
    params = list(range(1,5,1))
    best_param = 0
    best_accuracy = 0
    for param in params:
        print('CURRENT PARAM: ', param)
        model = train_supervised(
            input=train_data, epoch=8, lr=0.1, dim=120,
            wordNgrams=6, verbose=2, ws=8, lrUpdateRate=100,
        )
        correct = 0
        total = 0
        with open(valid_data, 'r') as f:
            for example in f.readlines():
                label = example.split(' ')[0]
                text = example.split(' ')[1:]
                text = ' '.join(text).replace('\n', '')
                pred = model.predict(text)
                if pred[0][0] == label:
                    correct += 1
                total += 1
        print_results(*model.test(valid_data))
        accuracy = correct/total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = param
        #model.save_model("mobile_extractions_model{}.bin".format(param))
    print(f'Best accuracy {best_accuracy}')
    print(f'Best param: {best_param}')
        

    #model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    #print_results(*model.test(valid_data))
    #model.save_model("title.ftz")

