import json
import os
from pathlib import Path

#Edit this gordon if you saved the json in a diff path windows: r"C:\path\to\translations\json"
PATH_TO_TRANSLATIONS_MAPPING_JSON = '/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data_utils/translations_mapping.json'
#Edit this with the language, usually 2 letter combination eg: en for english. language mapping is in word_to_lang.json
LANGUAGE_TO_REMOVE = ''

#Below can be unchanged
PATH_TO_WORD_2_LANG_JSON = Path.cwd() / 'word_to_lang.json'
OUTPUT_JSON = Path(PATH_TO_TRANSLATIONS_MAPPING_JSON).parent / 'translations_mapping_output.json'

with open(PATH_TO_TRANSLATIONS_MAPPING_JSON, 'r') as f:
    translations_mapping = json.load(f)
with open(PATH_TO_WORD_2_LANG_JSON, 'r') as f:
    word2lang = json.load(f)

new_mapping = {}
for word, translated_word in translations_mapping.items():
    if word in word2lang.keys():
        language = word2lang[word]
        if language == LANGUAGE_TO_REMOVE:
            print(f'Word:{word}, Translation: {translated_word} will be removed')
            continue
        else:
            new_mapping[word] = translated_word
    else:
        raise LookupError(f'Unrecognised word {word}, shouldnt be happening')

with open(OUTPUT_JSON, 'w') as f:
    f.write(json.dumps(new_mapping))
