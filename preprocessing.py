import pandas as pd
import numpy as np
import re
from math import log
from collections import defaultdict

def tokenize(text):
    translation_table = str.maketrans({c: f' {c} ' if not c.isalnum() else c for c in set(text)})  # creates translation
    # table(dictionary) with the built-in function maketrans, set(text) makes an unordered collection of unique
    # elements through set comprehension, a concise way to create sets
    tokenized_text = text.translate(translation_table)  # uses translation table to add whitespace around special
    # characters and punctuation
    return tokenized_text.strip().lower().split()  # split-tokens split on space, lower-tokens made all lowercase,
    # strip-and leading or trailing whitespaces are removed from string

def calculate_tfidf(token, document, all_tokens):
    tf = document.count(token) / len(document) # 1+log(#oftimestokenindoc/total#termsindoc)
    idf = log(1 + len(tokens) / tokens.count(token)) # log(#ofdocs/#oftimestokenappearsincollection)
    tfidf = tf * idf
    return tfidf

custom_headers = ['Emotions', 'Text']
df = pd.read_excel('isear-test.xlsx', skiprows=1, header=None, names=custom_headers)

#extract dict
text = ''.join(df['Text'].astype(str))
tokens = tokenize(text) #all tokens, including repeating
vocab = set(tokens)
print(vocab)

# Define emotion labels, there are 7 different labels in the data
emotion_labels = ['joy', 'anger', 'guilt', 'fear', 'sadness', 'shame', 'disgust']

#make tf-idf sentence representations and output file
tfidf_dict = defaultdict(dict)
for index, row in df.iterrows():
    emotion = row['Emotions']
    text = row['Text']
    tokenized_text = tokenize(text)
    tfidf_rep = defaultdict(float)
    for token in set(tokenized_text):
        tfidf_rep[token] = calculate_tfidf(token, tokenized_text, tokens)
    tfidf_dict[emotion][index] = tfidf_rep

# Create a new DataFrame to store TF-IDF representations
tfidf_df = pd.DataFrame(columns=['Emotions', 'Text'])
dfs = []

# Populate the DataFrame with TF-IDF representations
for emotion, text_dict in tfidf_dict.items():
    for index, tfidf_rep in text_dict.items():
        temp_df = pd.DataFrame({'Emotions': [emotion], 'Text': [tfidf_rep]})
        dfs.append(temp_df)

# Concatenate all DataFrames in the list
tfidf_df = pd.concat(dfs, ignore_index=True)

# Write the DataFrame to a new Excel file
tfidf_df.to_excel('tfidf_representations.xlsx', index=False)

