import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pathlib import Path
from multiprocessing import cpu_count

psychic_learner_dir = Path.cwd().parent
data_dir = psychic_learner_dir / 'data'

with open(str(data_dir / 'titles.txt'), 'r') as f:
    sentences = f.read().splitlines()
print(len(sentences))
for i, sentence in enumerate(sentences):
    sentences[i] = simple_preprocess(sentence)

model = Word2Vec(sentences=sentences, size=100, ns_exponent=-0.5, workers=cpu_count())
model.save("word2vec.bin")

from gensim.models.keyedvectors import KeyedVectors
#model.wv.save_word2vec_format('word2vec.bin', binary=True)
