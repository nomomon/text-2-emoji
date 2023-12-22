import numpy as np

def calculate_delta(y_pred, y_true):
    # Assume delta is the mean absolute error (MAE) between predictions and true values
    return np.mean(np.abs(y_pred - y_true))

def bootstrap(y_pred, y_true, num_samples=1000):
    # Calculate observed delta
    observed_delta = calculate_delta(y_pred, y_true)
    
    # Initialize variables
    s = 0
    
    # Number of data points
    n = len(y_pred)
    
    for i in range(num_samples):
        # Draw a bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample_pred = y_pred[indices]
        bootstrap_sample_true = y_true[indices]
        
        # Calculate delta for the bootstrap sample
        delta_i = calculate_delta(bootstrap_sample_pred, bootstrap_sample_true)
        
        # Check if delta_i is greater than or equal to 2 * observed_delta
        if delta_i >= 2 * observed_delta:
            s += 1
    
    # Calculate p-value
    p_value = s / num_samples
    
    return p_value

