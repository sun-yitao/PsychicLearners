import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from nltk.tokenize import word_tokenize
import string

data_dir = Path.cwd().parent / 'data'
test_dir = data_dir / 'image' / 'original' / 'fashion_image'

for image in test_dir.glob('*.jpg'):
    extract = pytesseract.image_to_string(Image.open(image)).lower()
    extract = re.sub(r'\d +', '', extract)
    result = extract.translate(str.maketrans(""), string.punctuation)
    print(word_tokenize(result))
    #print(pytesseract.image_to_data(Image.open(image),output_type=pytesseract.Output.DATAFRAME)['conf'])
    
