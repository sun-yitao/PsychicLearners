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

if __name__ == "__main__":
    psychic_learners_dir = os.path.split(os.getcwd())[0]
    train_data = os.path.join(psychic_learners_dir, 'data', 'beauty_train_split.txt')
    valid_data = os.path.join(psychic_learners_dir, 'data', 'beauty_valid_split.txt')
    #u_model = train_unsupervised(train_data, model = 'skipgram', lr = 0.05, dim = 15, ws = 5, 
    #    epoch = 5, neg = 5, wordNgrams = 4, loss = 'ns', bucket = 2000000,
    #    lrUpdateRate = 100, t = 0.0001, label = '__label__', verbose = 2, pretrainedVectors = '')
    #u_model.save_model("fil9.vec")
    params = [1,2,3,4,5]
    for param in params:
        print(param)
        model = train_supervised(
            input=train_data, epoch=6, lr=0.1, dim=100,
            wordNgrams=4, verbose=2, ws=5, lrUpdateRate=100,
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
        model.save_model("beauty_model{}.bin".format(param))

            

    
    #model.save_model("title.bin")

    #model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    #print_results(*model.test(valid_data))
    #model.save_model("title.ftz")

