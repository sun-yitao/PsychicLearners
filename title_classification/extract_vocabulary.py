import os
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter


data_directory = os.path.join(os.path.split(os.getcwd())[0], 'data')
BIG_CATEGORY = 'fashion'
train = pd.read_csv(os.path.join(data_directory, 'train_split.csv'))


#Split into categories
def split_categories(train):
    category_split = []
    for i in range(1,58):
        row_of_is = train.loc[train['Category'] == i]
        title, category = row_of_is['title'], row_of_is['Category']
        title_words = [word for line in title for word in line.split()] #split into words
        category_split.append(title_words)
    return pd.DataFrame(category_split)

#Return titles that repeat
def get_non_unique_titles(all_titles):
    train_set = (all_titles)
    word_counter = Counter()
    for s in train_set:
        word_counter.update(s.split())

    not_unique_titles = {k: v for k, v in word_counter.items() if v > 1}
    return not_unique_titles

#returns a list of word counters for each category
def get_word_counters(category_split):
    word_counters = []
    for titles in category_split:
        train_set = (titles)
        word_counter = Counter()
        for s in train_set:
            word_counter.update(s.split())
        word_counters.append(word_counter)
    return word_counters



category_split = split_categories(train)
print(category_split)

#Within each class find unique titles and then add them into one list
title_all_classes = []
for titles in category_split:
    train_set = (titles)
    word_counter = Counter()
    for s in train_set:
        word_counter.update(s.split())

    title_all_classes = title_all_classes + list(word_counter)


#find non unique titles
not_unique_titles = get_non_unique_titles(title_all_classes)


print(not_unique_titles)
print("Number of non unique titles " + str(len(not_unique_titles)))

all_titles = train["title"]
title_words = pd.DataFrame([word for line in all_titles for word in line.split()] )
print("Number of titles " + str(len(title_words)))
# delete non unique titles
unique_category_split = []
for titles in category_split:
    unique_category_split.append(titles[titles[0].isin(not_unique_titles)])

print("Done with cleaning")
print("Number of titles " + str(sum(map(len, unique_category_split))))

print("All unique titles")
print(unique_category_split)

#Get word counts for unique titles
category_split = split_categories(train)
word_counters = get_word_counters(category_split)


#for counter in word_counters:
    #print(counter)


