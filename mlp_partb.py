# ============================================================
# PART B: MLP FROM SCRATCH USING NUMPY
# Breast Cancer Classification
# ============================================================

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert target to column vector
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# ============================================================
# 4. DEFINE ACTIVATION FUNCTION
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Derivative of sigmoid
def sigmoid_derivative(a):
    return a * (1 - a)


# ============================================================
# 5. INITIALIZE WEIGHTS AND BIASES
# ============================================================

np.random.seed(42)

input_size = 30
hidden_size = 10
output_size = 1

learning_rate = 0.01
epochs = 1000


W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


# ============================================================
# 6. TRAINING
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # FORWARD PROPAGATION
    # --------------------------------------------------------

    # Hidden layer
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)


    # --------------------------------------------------------
    # LOSS CALCULATION
    # --------------------------------------------------------

    epsilon = 1e-8

    loss = -np.mean(
        y_train * np.log(A2 + epsilon)
        + (1 - y_train) * np.log(1 - A2 + epsilon)
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    # Output layer error
    dZ2 = A2 - y_train

    # Gradient for W2
    dW2 = np.dot(A1.T, dZ2) / len(X_train)

    # Gradient for b2
    db2 = np.mean(dZ2, axis=0, keepdims=True)


    # Hidden layer error
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * sigmoid_derivative(A1)

    # Gradient for W1
    dW1 = np.dot(X_train.T, dZ1) / len(X_train)

    # Gradient for b1
    db1 = np.mean(dZ1, axis=0, keepdims=True)


    # --------------------------------------------------------
    # UPDATE WEIGHTS AND BIASES
    # --------------------------------------------------------

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1


    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch + 1, "Loss:", loss)


# ============================================================
# 7. PREDICTION
# ============================================================

# Forward propagation on test data

Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)


# Convert probabilities to classes
y_pred = (A2_test >= 0.5).astype(int)


# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Test Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)
