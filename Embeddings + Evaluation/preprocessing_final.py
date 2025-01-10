import pandas as pd # allows using DataFrames
import numpy as np # numerical operations
import string # string operations
from math import log # need log for tf-idf
from scipy.sparse import csr_matrix # efficient way to create sparse matrices

def tokenize_and_clean(text):
    translation_table = str.maketrans({c: f' {c} ' if not c.isalnum() else c for c in set(text)}) # uses a translation table to surround each non-alphanumeric character with spaces
    tokenized_text = text.translate(translation_table).strip().lower().split() # strips leading and trailing spaces, converts to lowercase, spilts into tokens
    clean_tokens = [token for token in tokenized_text if # filters out punctuation and digits
                    not all(char in string.punctuation for char in token) and not token.isdigit()]
    return clean_tokens


def calculate_tfidf(token, document, tokenized_text, collection):
    tf = document.count(token) / len(document) # how often term is in sentence
    idf = log(1 + (len(tokenized_text) / (collection.count(token) + 1))) # how many doc contain term
    return tf * idf


def preprocess_data(file_path, label_encoding):
    custom_headers = ['Emotions', 'Text'] # defines column names for DataFrame
    df = pd.read_excel(file_path, skiprows=1, header=None, names=custom_headers) # skips first row, uses custom headers
    y = df['Emotions'].values # extracts emotion labels
    y_encoded = np.array([label_encoding[label] for label in y]) # encodes each label as a number
    tokenized_text = [tokenize_and_clean(sentence) for sentence in df['Text'].astype(str)] # calls the tokenize_and_clean function
    return df, y_encoded, tokenized_text


def create_tfidf_matrix(tokenized_text, collection, term_to_col):
    tfidf_scores_list = [] # list to store scores for each sentence
    for sentence in tokenized_text: # iterates through
        tfidf_scores = {token: calculate_tfidf(token, sentence, tokenized_text, collection) for token in set(sentence)} # calls tf-idf function for each token
        tfidf_scores_list.append(tfidf_scores)

    n_docs = len(tfidf_scores_list) # number of docs
    rows, cols, data = [], [], [] # intializes sparse matrix
    for doc_idx, scores in enumerate(tfidf_scores_list): # iterates through scores
        for term, tfidf in scores.items():
            if term in term_to_col: # checks to see if term in vocab
                col_idx = term_to_col[term]
                rows.append(doc_idx)
                cols.append(col_idx)
                data.append(tfidf) # appends tfidf score

    return csr_matrix((data, (rows, cols)), shape=(n_docs, len(term_to_col))) # returns sparse tf-idf matrix


# the label encoding dictionary
label_encoding = {'anger': 0, 'disgust': 1, 'fear': 2, 'guilt': 3, 'joy': 4, 'sadness': 5, 'shame': 6}

# 1. Choose if you would like to preprocess training data
train_df, y_train_encoded, tokenized_text_train = preprocess_data('isear-train.xlsx', label_encoding)

# creates vocabulary and collection from training data
vocab = set(token for sentence in tokenized_text_train for token in sentence) # terms
collection = [token for instance in tokenized_text_train for token in instance] # all tokens

# creates term to column index mapping
term_to_col = {term: idx for idx, term in enumerate(sorted(vocab))}

# calls function to create TF-IDF matrix for training data
X_tfidf_sparse = create_tfidf_matrix(tokenized_text_train, collection, term_to_col)

# 2. Choose whether you would like to preprocess test data, eval data, or both.
test_df, y_test_encoded, tokenized_text_test = preprocess_data('isear-test.xlsx', label_encoding)
#test_df, y_test_encoded, tokenized_text_test = preprocess_data('isear-validation.xlsx', label_encoding)

# creates TF-IDF matrix for test data
X_tfidf_sparse_test = create_tfidf_matrix(tokenized_text_test, collection, term_to_col)
