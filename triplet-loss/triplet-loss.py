import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Computes the Triplet Loss for embedding ranking.
    
    Args:
        anchor: array-like of shape (N, D) or (D,)
        positive: array-like of shape (N, D) or (D,)
        negative: array-like of shape (N, D) or (D,)
        margin: float, margin parameter (m)
        
    Returns:
        float: the scalar mean loss across the batch
    """
    # Convert inputs to numpy arrays
    a = np.array(anchor)
    p = np.array(positive)
    n = np.array(negative)
    
    # Calculate the squared Euclidean distance
    # axis=-1 handles both (D,) and (N, D) arrays 
    d_ap = np.sum((a - p)**2, axis=-1)
    d_an = np.sum((a - n)**2, axis=-1)
    
    # Calculate the loss for each triplet
    losses = np.maximum(0.0, d_ap - d_an + margin)
    
    # Return the mean loss across the batch (or the scalar itself if 1D)
    return np.mean(losses)