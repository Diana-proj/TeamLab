# Directions to run the .py code:

### Clone the Repository

Go to terminal, enter the following commands and make sure that you choose a path to clone the repository to.

`git clone https://github.com/Diana-proj/TeamLab.git C:/Users/User/Path`

`cd C:/Users/User/Path`

### Create a Virtual Environment:

Once the files are downloaded, check to make sure they are in the folder and then run the following commmands.

`python -m venv venv`  # or `python3` if you're using Python 3.x

### Activate the Virtual Environment:

Unix/macOS: `source venv/bin/activate`

On Windows: `.\venv\Scripts\activate`

### Install Requirements:

`pip install -r requirements.txt`

The environment should be set up, simply open the files you would like to run to get started.

# Directions to run the .pynb code:

# Our Task: Emotion Recognition

## Baseline: Logistic Regression
- F1 score on Test: 0.583
- F1 score on Dev: 0.582

Data:
- Training Data: isear-train.xlsx
- Testing Data: isear-test.xlsx
- Validation Data: isear-validation.xlsx

Data preprocessing: preprocessing_final.py or preprocessing_final.ipynb
- this file does feature extraction with log-tf-idf, and LR_final uses variables saved from this file

Model: LR_final.py or LR_final.ipynb
- trains on training data, 5000 epochs, 0.02 learning rate
- tests on validation data or test data

Evaluation: evaluation.py
- compares true labels to predicted labels for 4 different files, the test file, validation file, the dummy classifier file, and the given predicted labels, isear_val_predictions.xlsx

Dummy Classifier for testing evaluation script: dummy_classifier.py or dummy_classifier.ipynb
-outputs excel file to be used in evaluation script




## Advanced Methods: 

Data:
- Training Data: isear-train.xlsx
- Testing Data: isear-test.xlsx
- Validation Data: isear-validation.xlsx

Data Preprocessing:
- experimented with BERT Embeddings
- found RoBERTA embeddings to be better
- tried out TF-IDF for our BiLSTM model

FNN:
- F1 score on Test:
- F1 score on Dev:


BiLSTM: bilstm_final.py and bilstm_tfidf.py
(RoBERTA Embeddings)
F1 score on Test: 0.604
F1 score on Dev: 0.598
(TF-IDF vectors)
F1 score on Test: 0.604
F1 score on Dev: 0.598

RoBERTA: 
F1 score on Test:
F1 score on Dev:


Data Instance Error Analysis:
- looked at which labels are more difficult to predict and what data instances are incorrectly predicted

Additional Features:
- added additional vocabulary to certain emotions to improve prediction on those emotions that are poorly predicted

Evaluation:
- built-in F1 score
