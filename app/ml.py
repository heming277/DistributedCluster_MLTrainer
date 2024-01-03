# ml.py

# Import necessary libraries for each ML framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.celery_worker import celery
import numpy as np

# Define a function to train a model using PyTorch
@celery.task
def train_pytorch_model(data, model_params):
    # Example model: simple feedforward neural network

    X_train, y_train = data
    input_size = model_params['input_size']
    hidden_size = model_params['hidden_size']
    output_size = model_params['output_size']
    learning_rate = model_params['learning_rate']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    
    # Define the model using the parameters
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    
    model = SimpleNet(input_size, hidden_size, output_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
   
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate)

    # Training loop
    for _ in range(epochs):  # Example: 10 epochs
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

   

    def calculate_accuracy(model, data_loader):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients for validation
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct / total

    val_accuracy = calculate_accuracy(model, val_loader)
    # Calculate training accuracy
    train_accuracy = calculate_accuracy(model, train_loader)

    torch.save(model.state_dict(), 'model.pth')
    # Return a success message or any relevant results
    return {
        "message": "PyTorch model trained successfully",
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy 
    }

# Define a function to train a model using TensorFlow
@celery.task
def train_tensorflow_model(data, model_params):
    # Unpack the data
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val'] #validation data
    
    # Unpack model parameters
    input_shape = model_params['input_shape']
    num_classes = model_params['num_classes']
    learning_rate = model_params['learning_rate']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    
    # Define the TensorFlow model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Convert the data to numpy arrays if they aren't already
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    train_accuracy = history.history['accuracy'][-1]  # Get the last training accuracy

    
    model.save('model.h5')
    
    # Return the validation loss and accuracy
    return {
        "message": "TensorFlow model trained successfully",
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "train_accuracy": train_accuracy,
        "history": history.history
    }

# Define a function to train a model using scikit-learn
@celery.task
def train_sklearn_model(data, model_params):
    # Unpack the data provided by the user
    X, y = data['X'], data['y']
    
    # Unpack model parameters
    # Here we use .get() to provide default values if a parameter is not supplied
    test_size = model_params.get('test_size', 0.2)
    random_state = model_params.get('random_state', None)
    max_iter = model_params.get('max_iter', 100)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize the Logistic Regression model with parameters
    model = LogisticRegression(max_iter=max_iter)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the validation data
    y_pred = model.predict(X_val)
    # Predict on the training data
    y_train_pred = model.predict(X_train)
    
    # Calculate the accuracy on the validation data
    accuracy = accuracy_score(y_val, y_pred)

    #Calculate the accuracy on the training data
    train_accuracy = accuracy_score(y_train, y_train_pred)
 

    from joblib import dump
    dump(model, 'logistic_regression_model.joblib')
    
    # Return the accuracy and a success message
    return {
        "message": "scikit-learn model trained successfully",
        "train_accuracy": train_accuracy,
        "accuracy": accuracy
    }

# A function to determine which framework to use based on user input
def train_model(framework, data, model_params):
    if framework == 'pytorch':
        return train_pytorch_model(data, model_params)
    elif framework == 'tensorflow':
        return train_tensorflow_model(data, model_params)
    elif framework == 'sklearn':
        return train_sklearn_model(data, model_params)
    else:
        raise ValueError("Unsupported framework specified")
