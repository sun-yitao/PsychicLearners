import os
from pathlib import Path
import numpy as np
"""converts bert output probabilities to .npy files"""
psychic_learners_dir = Path.cwd().parent
tsv_dir = psychic_learners_dir / 'data' / 'tsvs' / 'bert_output'
prob_dir = psychic_learners_dir / 'data' / 'probabilities'
categories = ['beauty', 'fashion', 'mobile']
for category in categories:
    val_path = tsv_dir / category / 'valid' / 'test_results.tsv'
    test_path = tsv_dir / category / 'test' / 'test_results.tsv'
    val_prob = np.loadtxt(str(val_path), delimiter='\t', ndmin=0)
    test_prob = np.loadtxt(str(test_path), delimiter='\t', ndmin=0)
    print(val_prob.shape)
    print(test_prob.shape)
    os.makedirs(str(prob_dir / category / 'bert_v2'), exist_ok=True)
    np.save(str(prob_dir / category / 'bert_v2' / 'valid.npy'), val_prob)
    np.save(str(prob_dir / category / 'bert_v2' / 'test.npy'), test_prob)
