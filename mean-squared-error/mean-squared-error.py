import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes the Mean Squared Error between true and predicted values.
    """
    # Convert inputs to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure shapes match exactly; return None if there is a mismatch
    if y_true.shape != y_pred.shape:
        return None
        
    # Compute the squared differences and then the mean
    error = np.mean((y_true - y_pred) ** 2)
    
    # Return as a standard Python float
    return float(error)