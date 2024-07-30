import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the BiLSTM model. The hyperparameters are set inside the main body of code.

        Args:
            input_size (int): Dimension of input features.
            hidden_size (int): Size of features in the hidden state.
            output_size (int): Size of output classes.
            num_layers (int): Number of recurrent layers.
        """
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because it's bidirectional

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

    def train_model(self, train_loader, val_loader, device, num_epochs, patience=3):
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            device (torch.device): Device to run the model on, chooses GPUs or CPUs
            num_epochs (int): Number of epochs to train, defined in main body of code.
            patience (int): Number of epochs to wait for early stopping, 3.

        Returns:
            list: Lists containing the training and validation loss history, to be plotted after training.
        """
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # Adds L2 regularization
        self.train()

        loss_history = []
        val_loss_history = []

        best_val_loss = float('inf') # for early stopping
        patience_counter = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = self(inputs.unsqueeze(1))  # Add time dimension
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).long()
                    outputs = self(inputs.unsqueeze(1))  # Add time dimension
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_history.append(val_loss)
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")

            # Early Stopping, stops model before it overfits
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        return loss_history, val_loss_history

    def predict(self, X_test_tensor, device):
        """
        Make predictions with the model.

        Args:
            X_test_tensor (torch.Tensor): Input tensor for the test data.
            device (torch.device): Device to run the model on.

        Returns:
            torch.Tensor: Predictions for the input data.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(X_test_tensor.unsqueeze(1).to(device))  # Add time dimension
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu()

    def eval_model(self, eval_loader, device, dataset_name):
        """
        Evaluate the model on a given dataset.

        Args:
            eval_loader (DataLoader): DataLoader for the evaluation data.
            device (torch.device): Device to run the model on.
            dataset_name (str): Name of the dataset for logging.

        Returns:
            tuple: Loss and F1 score for the evaluation data.
        """
        self.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = self(inputs.unsqueeze(1))  # Add time dimension
                loss += criterion(outputs, labels).item()
                _, predictions = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        loss /= len(eval_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f'[{dataset_name}] Loss: {loss}, F1 Score: {f1}')

        # Print classification report, gives f score for each emotion label
        report = classification_report(all_labels, all_predictions, target_names=label_encoding.keys())
        print(report)

        return loss, f1


def get_strings(dataframe):
    """
    Extract text strings from a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        list: List of text strings.
    """
    return dataframe['text'].tolist()


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load files
custom_headers = ['emotion', 'text']
train_data = pd.read_excel('isear-train.xlsx', skiprows=1, header=None, names=custom_headers)
test_data = pd.read_excel('isear-test.xlsx', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_excel('isear-validation.xlsx', skiprows=1, header=None, names=custom_headers)

# Extract labels
label_encoding = {'anger': 0, 'disgust': 1, 'fear': 2, 'guilt': 3, 'joy': 4, 'sadness': 5, 'shame': 6}
y = train_data['emotion'].values
y_train = np.array([label_encoding[label] for label in y])
y_train = torch.tensor(y_train, dtype=torch.long)

y = test_data['emotion'].values
y_test = np.array([label_encoding[label] for label in y])
y_test = torch.tensor(y_test, dtype=torch.long)

y = dev_data['emotion'].values
y_dev = np.array([label_encoding[label] for label in y])
y_dev = torch.tensor(y_dev, dtype=torch.long)

# Ensure the labels and texts are paired correctly
assert len(y_train) == len(train_data['text'])
assert len(y_test) == len(test_data['text'])
assert len(y_dev) == len(dev_data['text'])

# Get strings from data to make TF-IDF embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Set max_features to a reasonable number for your dataset
train_tfidf = tfidf_vectorizer.fit_transform(train_strings).toarray()
dev_tfidf = tfidf_vectorizer.transform(dev_strings).toarray()
test_tfidf = tfidf_vectorizer.transform(test_strings).toarray()

# Convert TF-IDF embeddings to tensors
train_sentence_embeddings = torch.tensor(train_tfidf, dtype=torch.float32)
dev_sentence_embeddings = torch.tensor(dev_tfidf, dtype=torch.float32)
test_sentence_embeddings = torch.tensor(test_tfidf, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_sentence_embeddings, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dev_dataset = TensorDataset(dev_sentence_embeddings, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

test_dataset = TensorDataset(test_sentence_embeddings, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize and train the model
input_size = train_sentence_embeddings.shape[1]  # Set input size to the number of TF-IDF features
hidden_size = 264
output_size = 7
num_epochs = 20

model = BiLSTM(input_size, hidden_size, output_size)  # No dropout

# Uncomment these lines to train the model, and comment out the line below that loads the model
'''
loss_history, dev_loss_history = model.train_model(train_loader, dev_loader, device, num_epochs, patience=3)
torch.save(model.state_dict(), 'bilstm_model_tfidf1.pth')

# Plot the loss curve
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss')
plt.plot(range(1, len(dev_loss_history) + 1), dev_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss Curves')
plt.legend()
plt.show()
'''

# Load model, comment out to train model
model.load_state_dict(torch.load('bilstm_model_tfidf.pth'))
model.to(device)

# Evaluate the model
model.eval_model(train_loader, device, 'Train Dataset')
model.eval_model(dev_loader, device, 'Dev Dataset')
model.eval_model(test_loader, device, 'Test Dataset')
