import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Computes the Kullback-Leibler Divergence between two probability distributions.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    
    # 1. Create a mask to filter out indices where p == 0
    # This prevents calculating log(0) and naturally handles the 0 * log(0) = 0 limit.
    mask = p > 0
    
    p_safe = p[mask]
    q_safe = q[mask]
    
    # 2. Add epsilon to q to prevent division by zero or log(0) 
    # for cases where p > 0 but q is extremely small or exactly 0.
    q_safe = q_safe + eps
    
    # 3. Compute and sum the element-wise divergence
    kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
    
    return float(kl_div)