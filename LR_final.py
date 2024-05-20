import pandas as pd # for DataFrame structures
import numpy as np # numerical operations, arrays
import preprocessing_final # imports functions and variables from preprocessing.py
import joblib # allows saving model

class LogisticRegressionMulticlass: # defines a multiclass log regression
    def __init__(self, learning_rate=0.3, num_iterations=10000): # specifies a learning rate and # of iterations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None # weight matrix
        self.bias = None # bias matrix

    def softmax(self, z): # uses softmax to obtain prob between 0 and 1, with prob summing to 1, input z
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # subtracts max value in each row before taking exponential to avoid numbers getting too big
        #keepdims=True ensures that dimensions stay the same
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # normalizes values by dividing the sum from each row, makes all rows sum to zero

    def xavier_init(self, shape): # intializes weights using Xavier initialization
        fan_in = shape[0] # number of input units to the layer
        fan_out = shape[1] # number of output units to the layer
        limit = np.sqrt(6 / (fan_in + fan_out)) # returns matrix of weights initialized between the limits
        return np.random.uniform(-limit, limit, size=shape) # uniform distribution

    def fit(self, X, y):
        num_samples, num_features = X.shape # num_samples is number of rows in input data, num_features is number of columns
        num_classes = len(np.unique(y)) # number of unique labels in data
        self.weights = self.xavier_init((num_features, num_classes)) # calls xavier initialization
        self.bias = np.zeros((1, num_classes)) # initializes bias vector with zeros
        y_one_hot = np.eye(num_classes)[y] # creates identity matrix, converts labels into one-hot encoded vectors

        for _ in range(self.num_iterations): # training
            linear_model = X.dot(self.weights) + self.bias # computes dot product
            y_pred = self.softmax(linear_model) # applys softmax

            dw = (1 / num_samples) * X.T.dot(y_pred - y_one_hot) # computes gradient for weights and biases
            db = (1 / num_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw # updates weights by subtracting product of learning rate and gradient dw
            self.bias -= self.learning_rate * db # updates biases

    def predict(self, X): # predicts labels for input data X

        num_samples, num_features = X.shape # retrieves number of features and samples from X
        _, num_classes = self.weights.shape # retrives the number of output classes from the weights matrix

        if num_features != self.weights.shape[0]: # checks that num of features in input data matches number of features expected by weights matrix
            self.weights = self.xavier_init((num_features, num_classes)) # if they do not match, reintializes weights

        linear_predictions = X.dot(self.weights) + self.bias # matrix multiplication between X and weights, results in matrix with shape [numsamples, numclasses]
        # +bias adds the bias term to each row of resulting matric
        y_pred = self.softmax(linear_predictions) # applys softmax to generate prob
        class_pred = np.argmax(y_pred, axis=1)  # chooses the class with the highest probability
        return class_pred # returns an array where each element is predicted class for corresponding sample


model_emotions = LogisticRegressionMulticlass() # creates an instance of the LogisticRegressionMulticlass
model_emotions.fit(preprocessing_final.X_tfidf_sparse, preprocessing_final.y_train_encoded) # trains the model

# 1. Choose whether to save the trained model
joblib.dump(model_emotions, 'logistic_regression_model.pkl')

# 2. If you have already saved a model, remove the # to load your trained model and comment out the saving model line
#loaded_model = joblib.load('logistic_regression_model.pkl')

y_external_pred = model_emotions.predict(preprocessing_final.X_tfidf_sparse_test) # predicts labels

emotion_mapping = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'guilt', 4: 'joy', 5: 'sadness', 6: 'shame'} # reverse emotion mapping to retrieve labels from numbers
pred_data = np.array([emotion_mapping[value] for value in y_external_pred])

pred_df = pd.DataFrame({'Predicted Emotions': pred_data}) # writes predictions along with their corresponding text data to a file

# 3. Choose the file you want to make predictions on, eval or test file
pred_df.to_excel("predictions_with_test.xlsx", index=False)
#pred_df.to_excel("predictions_with_eval.xlsx", index=False)




