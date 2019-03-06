from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from fastText import train_supervised
import numpy as np


def print_results(N, p, r):
    print("N\t" + str(N))
    print("Accuracy: {}".format(correct/N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == "__main__":
    psychic_learners_dir = os.path.split(os.getcwd())[0]
    train_data = os.path.join(psychic_learners_dir, 'data', 'beauty_train_split.txt')
    valid_data = os.path.join(psychic_learners_dir, 'data', 'beauty_valid_split.txt')
    model = train_supervised(
        input=train_data, epoch=10, lr=0.1, dim=20,
        wordNgrams=4, verbose=2, ws=7, lrUpdateRate=100

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
    print_results(*model.test(valid_data))
            

            

    
    #model.save_model("title.bin")

    #model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    #print_results(*model.test(valid_data))
    #model.save_model("title.ftz")

