import nltk
import re

def convertlist(filename):
    list_out = list()
    fhandle = open(filename)
    for line in fhandle.readlines():
        line = line.replace("\n", "").strip()
        line = line.lower()
        try:
            texts_tokenized = word_tokenize(line)
            for words in texts_tokenized:
                english_punctuations = [
                    ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
                if not words in english_punctuations:
                    list_out.append(words)
        except:
            continue
    return list_out


def get_common_words():
    ret_list = []
    out_path = "./sample_data"
    content = convertlist(out_path)
    fdist2 = nltk.FreqDist(content)
    most_list = fdist2.most_common(400)
    for x, value in most_list:
        ret_list.append(x)
    return ret_list

def clean_words():
    result = re.sub(r’\d +’, ‘’, input_str)


def main():
	common_list = get_common_words()
	print common_list[0:100]


if __name__ == '__main__':
   		main()
