import numpy as np

def _sigmoid(z):
    """
    Numerically stable sigmoid function.
    Clips values to avoid overflow in np.exp()
    """
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, steps: int):
    """
    Trains a binary logistic regression classifier using gradient descent.
    
    Args:
        X: numpy array of shape (N, D) containing the training data
        y: numpy array of shape (N,) containing the binary labels (0 or 1)
        lr: learning rate (float)
        steps: number of gradient descent iterations (int)
        
    Returns:
        tuple (w, b) where w is a numpy array of shape (D,) and b is a float
    """
    # 1. Initialize parameters based on input dimensions
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    
    # 2. Training loop
    for _ in range(steps):
        # Forward pass: compute the linear combination and apply sigmoid
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # Compute the difference between predictions and actual labels
        error = p - y
        
        # Backward pass: compute gradients
        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)
        
        # Gradient descent step: update weights and bias
        w -= lr * dw
        b -= lr * db
        
    return w, b