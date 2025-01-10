import pandas as pd  # for DataFrames
import random  # for selecting at random

custom_headers = ['Emotions', 'Text']
df = pd.read_excel('isear-test.xlsx', skiprows=1, header=None, names=custom_headers)  # loads test file

total_count = df['Emotions'].value_counts().sum()
df['Emotions'].value_counts()  # counts occurances of emotions
emotions_set = list(set(df['Emotions'].values))  # defines a set of emotions as extracted from the test set


def random_classifier():  # randomly selects an emotion from emotions_set and returns it
    return random.choice(emotions_set)


df['Predicted_Emotion'] = df['Text'].apply(
    lambda x: random_classifier())  # calls random_classifier function for each line of DataFrame

df_subset = df[['Predicted_Emotion', 'Text']]  # makes new DataFrame with predicted emotions and original text

excel_file_path = "dummy_classifier.xlsx"  # saves random predictions to a new file
df_subset.to_excel(excel_file_path, index=False)
