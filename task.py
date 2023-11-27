from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from arl import AdaptiveResonanceLayer

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split the dataset into training and testing sets
# Create DataLoader for batch processing
# Neural Network Model
class ARLNet(nn.Module):
    def __init__(self):
        super(ARLNet, self).__init__()
        self.arl = AdaptiveResonanceLayer(20, 64)  # 20 features input, 64 units in ARL
        self.fc = nn.Linear(64, 1)  # Final output layer

    def forward(self, x):
        x = self.arl(x)
        x = torch.sigmoid(self.fc(x))
        return x

model = ARLNet()

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track the gradients
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()

            # Convert outputs to predictions
            predicted = outputs.round().squeeze()

            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy

from sklearn.model_selection import KFold
from itertools import product
import numpy as np

# Hyperparameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
}

# Function to create DataLoaders
def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# K-Fold Cross-Validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store the best hyperparameters and their performance
best_params = None
best_accuracy = 0

X_tensor = X
y_tensor = y


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Grid Search
for lr, batch_size in product(param_grid['learning_rate'], param_grid['batch_size']):
    print(f"Testing with lr={lr}, batch_size={batch_size}")
    avg_accuracy = 0

    for fold, (train_ids, test_ids) in enumerate(kf.split(X_tensor)):
        # Data preparation for this fold
        X_train, X_val = X_tensor[train_ids], X_tensor[test_ids]
        y_train, y_val = y_tensor[train_ids], y_tensor[test_ids]
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size)

        # Model initialization
        model = ARLNet()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epoch = 0

        early_stopper = EarlyStopping(patience=5, min_delta=0.001)

        # Training Loop
        while not early_stopper.early_stop:
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()

            early_stopper(loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            epoch += 1

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                predicted = outputs.round().squeeze()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_accuracy = correct / total
        avg_accuracy += val_accuracy

    # Average accuracy across all folds
    avg_accuracy /= 5

    # Update the best parameters if this combination is better
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params = {'learning_rate': lr, 'batch_size': batch_size}

# Print the best parameters found
print(f"Best Parameters: {best_params}, Best Average Validation Accuracy: {best_accuracy}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.clone().detach()
y_train = y_train.clone().detach()
X_test = X_test.clone().detach()
y_test = y_test.clone().detach()

# Train the model with the best hyperparameters on the entire training set
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)

model = ARLNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

early_stopper = EarlyStopping(patience=5, min_delta=0.001)

epoch = 0

while not early_stopper.early_stop:
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    early_stopper(loss)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

    epoch += 1

# Test the model on the test set
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=best_params['batch_size'], shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = outputs.round().squeeze()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_accuracy = correct / total
print(f'Test Accuracy with Best Hyperparameters: {test_accuracy}')