import numpy as np
from pathlib import Path

psychic_learners_dir = Path.cwd().parent
BIG_CATEGORY = 'beauty'
elmo_dir = psychic_learners_dir / 'data' / 'features' / BIG_CATEGORY / 'elmo'
valid = np.load(str(elmo_dir / 'valid.npy'), allow_pickle=True)
test = np.load(str(elmo_dir / 'test.npy'), allow_pickle=True)

new_valid = np.empty((len(valid), 16 * 1024))
new_test = np.empty((len(test), 16 * 1024))

for n, array in enumerate(valid):
    flattened = array.flatten()
    if len(flattened) > 16*1024:
        padded = flattened[:16*1024]
    else:
        padded = np.pad(flattened, (0, 16*1024 - len(flattened)), 'constant', constant_values=0)
    new_valid[n] = padded
print(new_valid.shape)
np.save(str(elmo_dir / 'valid_flat.npy'), new_valid)
for n, array in enumerate(test):
    flattened = array.flatten()
    if len(flattened) > 16*1024:
        padded = flattened[:16*1024]
    else:
        padded = np.pad(flattened, (0, 16*1024 - len(flattened)), 'constant', constant_values=0)
    new_test[n] = padded

print(new_test.shape)
np.save(str(elmo_dir / 'test_flat.npy'), new_test)
