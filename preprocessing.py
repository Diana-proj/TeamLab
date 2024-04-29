import pandas as pd
from math import log

def tokenize(text):
    translation_table = str.maketrans({c: f' {c} ' if not c.isalnum() else c for c in set(text)})
    tokenized_text = text.translate(translation_table)
    return tokenized_text.strip().lower().split()

def calculate_tfidf(token, document, all_tokens):
    tf = document.count(token) / len(document)
    idf = log(1 + len(all_tokens) / all_tokens.count(token))
    return tf * idf

custom_headers = ['Emotions', 'Text']
df = pd.read_excel('isear-test.xlsx', skiprows=1, header=None, names=custom_headers)

# Extract vocabulary
text = ''.join(df['Text'].astype(str))
tokens = tokenize(text)
vocab = set(tokens)

# Define emotion labels
emotion_labels = ['joy', 'anger', 'guilt', 'fear', 'sadness', 'shame', 'disgust']

# Create a list to store TF-IDF representations
tfidf_rows = []

# Calculate TF-IDF for each row and store it in the list
for index, row in df.iterrows():
    emotion = row['Emotions']
    text = row['Text']
    tokenized_text = tokenize(text)
    tfidf_rep = {token: calculate_tfidf(token, tokenized_text, tokens) for token in set(tokenized_text)}
    tfidf_rows.append({'Emotions': emotion, 'Text': tfidf_rep})

# Create DataFrame from the list of TF-IDF representations
tfidf_df = pd.DataFrame(tfidf_rows)

# Write the DataFrame to a new Excel file
tfidf_df.to_excel('tfidf_representations.xlsx', index=False)
