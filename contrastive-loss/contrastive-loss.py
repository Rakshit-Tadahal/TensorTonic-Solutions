import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    # Convert inputs to NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    # Validate y contains only 0 or 1
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels 'y' must strictly be 0 or 1.")

    # Handle both (D,) and (N, D) shapes safely
    if a.ndim == 1:
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
    
    # Ensure y broadcasts correctly (squeeze out extra dims, ensure at least 1D)
    y = np.atleast_1d(np.squeeze(y))

    # Calculate the Euclidean distance d for each pair (N,)
    d = np.linalg.norm(a - b, axis=1)

    # Compute loss using the vectorized formula
    # loss = y * d^2 + (1 - y) * max(0, m - d)^2
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    # Apply reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("Invalid reduction method. Choose 'mean' or 'sum'.")