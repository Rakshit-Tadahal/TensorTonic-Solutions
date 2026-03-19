import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Computes the Wasserstein Critic Loss for a WGAN.
    Formula: L = E[D(fake)] - E[D(real)]
    """
    # Convert inputs to NumPy arrays to leverage fast C-level operations
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    
    # Calculate the expected values (means) of both sets of scores
    expected_fake = np.mean(fake_scores)
    expected_real = np.mean(real_scores)
    
    # Compute the final critic loss
    loss = expected_fake - expected_real
    
    # Return as a standard Python float
    return float(loss)