import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(768, 256)  # 768 is BERT's embedding size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)# softmax for multi-class classification
        return out

    def train_model(self, train_loader, device, num_epochs):
        self.to(device)
        criterion = nn.CrossEntropyLoss()  # CrossEntropy fro multi-class
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()

        loss_history = []  # To store the loss at each epoch
        dev_loss_history = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = self(inputs)
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
                for inputs, labels in dev_loader:
                    inputs, labels = inputs.to(device), labels.to(device).long()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(dev_loader)
            dev_loss_history.append(val_loss)
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")

        return loss_history, dev_loss_history

    def predict(self, X_test_tensor, device):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test_tensor.to(device))
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu()

    def eval_model(self, eval_loader, device, dataset_name):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = self(inputs)
                loss += criterion(outputs, labels).item()
                _, predictions = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        loss /= len(eval_loader)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f'[{dataset_name}] Loss: {loss}, F1 Score: {f1}')
        return loss, f1


def get_sentence_embeddings(text_list, tokenizer, bert_model, device, batch_size=64):
    bert_model.eval()
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.pooler_output
        all_embeddings.append(batch_embeddings.cpu())
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    return sentence_embeddings


def get_strings(dataframe):
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
print("Encoded labels:", y, y_train[:20])
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

# Get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)
'''
# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Get embeddings
train_sentence_embeddings = get_sentence_embeddings(train_strings, tokenizer, bert_model, device)
dev_sentence_embeddings = get_sentence_embeddings(dev_strings, tokenizer, bert_model, device)
test_sentence_embeddings = get_sentence_embeddings(test_strings, tokenizer, bert_model, device)

# Save and load embeddings (if needed)
torch.save(train_sentence_embeddings, "train_sentence_embeddings.pt")
torch.save(dev_sentence_embeddings, "dev_sentence_embeddings.pt")
torch.save(test_sentence_embeddings, "test_sentence_embeddings.pt")

'''
train_sentence_embeddings = torch.load("train_sentence_embeddings.pt")
dev_sentence_embeddings = torch.load("dev_sentence_embeddings.pt")
test_sentence_embeddings = torch.load("test_sentence_embeddings.pt")

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_sentence_embeddings, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dev_dataset = TensorDataset(dev_sentence_embeddings, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

test_dataset = TensorDataset(test_sentence_embeddings, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize and train the model
model = FNN()
loss_history, dev_loss_history = model.train_model(train_loader, device, num_epochs=400)
torch.save(model.state_dict(), 'fnn_comp_model.pth')

# Plot the loss curve
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss')
plt.plot(range(1, len(dev_loss_history) + 1), dev_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss Curves')
plt.legend()
plt.show()

# load model
#model.load_state_dict(torch.load('fnn_model.pth'))
#model.to(device)

# Evaluate the model
model.eval_model(dev_loader, device,'Dev Dataset')
model.eval_model(test_loader, device, 'Test Dataset')


