import re
import json

def correct_word(word):
    if(re.search("\w\.(\w\.)+\w", word)):
        word = re.sub("\.+", "", word)
    word = re.sub("\.+", " ", word)
    word = re.sub("\++", " ", word)
    word = re.sub("\&+", " ", word)
    word = re.sub("\:+", " ", word)

    word = word.replace("www.", " ")
    word = word.replace(".com", " ")

    return word

filename_read = "alphabetic_misspelt_and_weird_mappings0.json"
filename_write = "fix_for_WEIRD_mappings.json"

with open(filename_read, 'r') as f:
    datastore = json.load(f)

for item in datastore:
    corrected_word = correct_word(item)
    print(corrected_word)
    datastore[item] = corrected_word

print(datastore)

with open(filename_write, 'w') as f:
    json.dump(datastore, f)