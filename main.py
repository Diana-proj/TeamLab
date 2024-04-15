import pandas as pd
import numpy as np

# Load the data from the Excel file
data = pd.read_excel("isear-test.xlsx")

# Defines a simple text preprocessing function
def preprocess_text(text):
    translation_table = str.maketrans({c: f' {c} ' if not c.isalnum() else c for c in set(text)})  # creates translation
    # table(dictionary) with the built-in function maketrans, set(text) makes an unordered collection of unique
    # elements through set comprehension, a concise way to create sets
    tokenized_text = text.translate(translation_table)  # uses translation table to add whitespace around special
    # characters and punctuation
    return tokenized_text.strip().lower().split()

# Apply text preprocessing to the 'text' column
data['text'] = data['text'].apply(preprocess_text)

# Define the SVM algorithm
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Split the data into features (X) and labels (y)
X = data['text'].values
y = data['emotion'].values

# Split the data into training and testing sets (you can implement this manually)
# For simplicity, let's say we use the first 80% of data for training and the remaining 20% for testing
split_idx = int(0.8 * len(data))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Initialize and train the SVM classifier
svm_classifier = SVM()
svm_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate accuracy (you can implement this manually)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
