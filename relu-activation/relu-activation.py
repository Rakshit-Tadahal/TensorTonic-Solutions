import numpy as np

def relu(x):
    """
    Computes the Rectified Linear Unit (ReLU) activation element-wise.
    Formula: ReLU(x) = max(0, x)
    """
    # np.maximum natively handles scalars, lists, and arrays element-wise
    # and automatically returns a NumPy array or NumPy scalar.
    return np.maximum(0, x)