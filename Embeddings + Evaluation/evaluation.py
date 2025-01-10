import pandas as pd # imports pandas lib and assigns it to the alias pd
# panda is built on top of NumPy, good for structured data, includes DataFrame, series, can help with cleaning data, files of various formats...

custom_headers = ['Emotions', 'Text'] # defines a list of column headers that will be used when loading the excel test or eval file

# 1. Choose either test or eval data to evaluate on:
test_data = pd.read_excel("isear-test.xlsx", skiprows=1, header=None, names=custom_headers) # retrieves true labels for test file
#eval_data = pd.read_excel("isear-validation.xlsx", skiprows=1, header=None, names=custom_headers) # true labels for eval file

# 2. Choose predicted label file accordingly:
pred_data = pd.read_excel("predictions_with_test.xlsx") # retrieve pred labels for test file
#pred_data = pd.read_excel('dummy_classifier.xlsx', skiprows=1, header=None, names=custom_headers) # retrieve pred labels generated from dummy classifier
#pred_data = pd.read_excel('isear-val-prediction.xlsx', skiprows=1, header=None, names=custom_headers) # retrieve pred labels from the given pred file
#pred_data = pd.read_excel("predictions_with_eval.xlsx") # retrieve pred labels on validation set

# 3. Choose test or eval data for true labels:
#labels = eval_data['Emotions'] # extracts emotion column from the test data
labels = test_data['Emotions']
predictions = pred_data['Predicted Emotions'] # extracts emotion column from prediction file

# now run code

# Define emotion labels, there are 7 different labels in the data
emotion_labels = ['joy', 'anger', 'guilt', 'fear', 'sadness', 'shame', 'disgust']

label_f_scores = {} # initializes a dictionary to store the precision, recall, and F-score for each emotion label
for emotion in emotion_labels: # loops over each emotion in emotion list
    true_positive = sum(labels.eq(emotion) & predictions.eq(emotion)) #calculates # of TP for current emotion label
    false_positive = sum(labels.ne(emotion) & predictions.eq(emotion))
    # .eq is used to compare two series or DataFrames, returns true if equal and false if not. .ne is != for strings
    false_negative = sum(labels.eq(emotion) & predictions.ne(emotion))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    #stores p, r, and f for the current emotion label in the dict with emotion label being the key
    label_f_scores[emotion] = {'Precision': precision, 'Recall': recall, 'F-score': f_score}

# Print F-score for each emotion label
print("F-score for each emotion label:")
for emotion, scores in label_f_scores.items():
    print(
        f"{emotion}: Precision: {scores['Precision']}, Recall: {scores['Recall']}, F-score: {scores['F-score']}")

# Calculate overall F-score
overall_precision = sum(scores['Precision'] for scores in label_f_scores.values()) / len(emotion_labels)
overall_recall = sum(scores['Recall'] for scores in label_f_scores.values()) / len(emotion_labels)
overall_f_score = sum(scores['F-score'] for scores in label_f_scores.values()) / len(emotion_labels)

# Print overall F-score
print("\nOverall F-score:")
print(f"F-score: {overall_f_score}")
