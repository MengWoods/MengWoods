# Introduction
# This script demonstrates the creation, training, and evaluation of a simple neural network using PyTorch.
# The synthetic data generated follows a specific rule based on the sum of its features, and the network is trained to classify this data.
# The script is modular, with functions for generating data, defining the model, training, and evaluation.
# The main function controls whether the script runs in training mode or evaluation mode.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Function to generate synthetic data with some underlying rule
def generate_synthetic_data(num_samples, input_dim, num_classes):
    inputs = torch.randn(num_samples, input_dim)
    targets = torch.empty(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        feature_sum = inputs[i].sum().item()
        # Assign target labels based on feature sum ranges
        if feature_sum < -10:
            targets[i] = 0
        elif feature_sum < -5:
            targets[i] = 1
        elif feature_sum < 0:
            targets[i] = 2
        elif feature_sum < 5:
            targets[i] = 3
        elif feature_sum < 10:
            targets[i] = 4
        elif feature_sum < 15:
            targets[i] = 5
        elif feature_sum < 20:
            targets[i] = 6
        elif feature_sum < 25:
            targets[i] = 7
        elif feature_sum < 30:
            targets[i] = 8
        else:
            targets[i] = 9
            
    return inputs, targets

# Define the neural network model class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Fully connected layer from 784 to 128 neurons
        self.fc2 = nn.Linear(128, 10)   # Fully connected layer from 128 to 10 neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = self.fc2(x)              # Output layer, no activation here
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()  # Zero the gradients before the backward pass
            outputs = model(batch_inputs)  # Forward pass: compute the output
            loss = criterion(outputs, batch_targets)  # Compute the loss
            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update weights
            running_loss += loss.item() * batch_inputs.size(0)  # Accumulate loss for monitoring
        epoch_loss = running_loss / len(dataloader.dataset)  # Calculate the average loss
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    print('Training complete.')

def evaluate_model(model, inputs):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during inference
        predictions = model(inputs)
    for i in range(len(inputs)):
        print(f"Input {i+1} Sum: {inputs[i].sum().item()}")
        print(f"Prediction {i+1}: {predictions[i].argmax().item()}")
        print("-" * 50)

def main(mode='train'):
    # Hyperparameters
    num_samples = 1000
    input_dim = 784
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 500
    
    # Generate synthetic data
    inputs, targets = generate_synthetic_data(num_samples, input_dim, num_classes)
    
    # Create a DataLoader for batching
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate the model
    model = SimpleNN()
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks with multiple classes
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer with learning rate of 0.001
    
    if mode == 'train':
        # Train the model
        train_model(model, dataloader, criterion, optimizer, num_epochs)
        # Save the model state dictionary
        torch.save(model.state_dict(), 'model.pth')
        print("Model saved to 'model.pth'")
    elif mode == 'eval':
        # Load the model
        model.load_state_dict(torch.load('model.pth'))
        print("Model loaded from 'model.pth'")
        # Example new input data
        new_inputs = torch.randn(10, 784)  # Batch of 10 new samples, each with 784 features
        # Evaluate the model
        evaluate_model(model, new_inputs)

if __name__ == '__main__':
    main(mode='eval')  # Set to 'train' or 'eval' based on your requirement
