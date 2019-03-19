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


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = text.strip().lower()
    return text

for image in test_dir.glob('*.jpg'):
    extract = pytesseract.image_to_string(Image.open(image))
    extract = extract.strip().lower()
    if not extract:
        continue
    extract = re.sub(r"[^A-Za-z0-9]", " ", extract)
    extract = re.sub(r'\d +', '', extract)
    extract = word_tokenize(extract)
    extract = [word for word in extract if word.isalnum() and len(word) > 2]
    print(' '.join(extract))
    #print(pytesseract.image_to_data(Image.open(image),output_type=pytesseract.Output.DATAFRAME)['conf'])
    
