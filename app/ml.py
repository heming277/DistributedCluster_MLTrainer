# app/ml.py

# Import necessary libraries for each ML framework
from flask import current_app
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


from app.celery_worker import celery
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler
import numpy as np
from joblib import dump
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns




# Define the models directory
models_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

if not os.path.exists(models_directory):
    os.makedirs(models_directory)

def plot_training_history(history, plot_path='static/training_history.png'):
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    sns.lineplot(data=history, x=range(len(history['train_accuracy'])), y='train_accuracy', label='Train', marker='o')
    sns.lineplot(data=history, x=range(len(history['val_accuracy'])), y='val_accuracy', label='Validation', marker='o')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    sns.lineplot(data=history, x=range(len(history['train_loss'])), y='train_loss', label='Train', marker='o')
    sns.lineplot(data=history, x=range(len(history['val_loss'])), y='val_loss', label='Validation', marker='o')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    

# Define a function to train a model using PyTorch
@celery.task
def train_pytorch_model(data, model_params):
    # Convert the 'Species' column to integers

    history = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': []
    }


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['y'])

    # Split the data into training and validation sets before converting to tensors
    X_train, X_val, y_train, y_val = train_test_split(
        np.array(data['X'], dtype=np.float32),
        y_encoded,
        test_size=0.2,
        random_state=42  
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert the training and validation data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)


     # Handling class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    samples_weights = torch.tensor([class_weights[i] for i in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))


    # Create TensorDatasets and DataLoaders for training and validation
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=False)

    # Define the model using the parameters
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = SimpleNet(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        output_size=model_params['output_size']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # Adjusted scheduler

    # Initialize variables for Early Stopping
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before stopping
    patience_counter = 0  # Tracks how many epochs have passed without improvement

    # Training loop
    for epoch in range(model_params['epochs']):
        model.train()  # Set the model to training mode
        train_loss_accumulated = 0
        total_train = 0
        correct_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_accumulated += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        avg_train_loss = train_loss_accumulated / len(train_loader)
        train_accuracy = correct_train / total_train
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss_accumulated = 0
        total_val = 0
        correct_val = 0
        with torch.no_grad():  # No need to track gradients for validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_accumulated += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        avg_val_loss = val_loss_accumulated / len(val_loader)
        val_accuracy = correct_val / total_val
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        scheduler.step(avg_val_loss)

        # Early Stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model
            best_model_path = os.path.join(models_directory, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break  # Exit the training loop

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Load the best model for further use or evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Return results, including the history
    return {
        'model_type': 'pytorch',
        "message": "PyTorch model trained successfully",
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "model_filename": 'best_model.pth',
        "history": history  # Include the history in the return statement
    }

def calculate_accuracy(model, data_loader, device):
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




# Define a function to train a model using TensorFlow
@celery.task
def train_tensorflow_model(data, model_params):
    # Unpack the data
    X, y = map(np.array, (data['X'], data['y']))
    
    # Encode the string labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Unpack model parameters
    input_shape = model_params.get('input_shape')
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    num_classes = len(np.unique(y))  # Update num_classes based on the unique labels
    learning_rate = model_params['learning_rate']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    

    # Normalize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Define the TensorFlow model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    train_accuracy = history.history['accuracy'][-1]  # Get the last training accuracy
    
    # Save the model
    model_save_path = os.path.join(models_directory, 'model.keras')
    model.save(model_save_path)
    
    # Return the validation loss and accuracy
    return {
        'model_type': 'tensorflow',
        "message": "TensorFlow model trained successfully",
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "train_accuracy": train_accuracy,
        "history": history.history,
        "model_filename": 'model.keras'  # Add the filename to the response
    }








# Define a function to train a model using scikit-learn
@celery.task
def train_sklearn_model(data, model_params):
    # Unpack the data provided by the user
    X, y = np.array(data['X']), np.array(data['y'])
    
    # Unpack model parameters
    test_size = model_params.get('test_size', 0.2)
    random_state = model_params.get('random_state', None)
    max_iter = model_params.get('max_iter', 100)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize and train the Logistic Regression model
    lr_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    lr_model.fit(X_train, y_train)
    lr_val_accuracy = accuracy_score(y_val, lr_model.predict(X_val))

    # Initialize the RandomForestClassifier with default parameters for RandomizedSearchCV
    rf_model = RandomForestClassifier(random_state=random_state)
    param_distributions = {
        'n_estimators': randint(50, 200),  # Number of trees in the forest
        'max_depth': randint(3, 10),  # Maximum depth of the tree
    }

    # Setup RandomizedSearchCV for the RandomForest model
    rf_search = RandomizedSearchCV(rf_model, param_distributions=param_distributions, n_iter=5, cv=3, random_state=random_state, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_rf_model = rf_search.best_estimator_
    rf_val_accuracy = accuracy_score(y_val, best_rf_model.predict(X_val))

    # Compare the validation accuracies and select the best model
    if lr_val_accuracy > rf_val_accuracy:
        best_model = lr_model
        best_accuracy = lr_val_accuracy
        model_type = 'LogisticRegression'
    else:
        best_model = best_rf_model
        best_accuracy = rf_val_accuracy
        model_type = 'RandomForestClassifier'

    # Save the best model
    model_save_path = os.path.join(models_directory, f'{model_type}.joblib')
    dump(best_model, model_save_path)

    # Return the accuracy and a success message
    return {
        'model_type': 'sklearn',
        "message": f"{model_type} model trained successfully",
        "train_accuracy": accuracy_score(y_train, best_model.predict(X_train)),
        "accuracy": best_accuracy,
        "model_filename": f'{model_type}.joblib'
    }




# A function to determine which framework to use based on user input
@celery.task
def train_model(framework, data, model_params):
    if framework == 'pytorch':
        return train_pytorch_model(data, model_params)
    elif framework == 'tensorflow':
        return train_tensorflow_model(data, model_params)
    elif framework == 'sklearn':
        return train_sklearn_model(data, model_params)
    else:
        raise ValueError("Unsupported framework specified")
