import numpy as np
import torch

def farthest_point_sampling(points, k):
    """Farthest Point Sampling (FPS) algorithm.
    
    Args:
        points (np.ndarray or torch.Tensor): The dataset to subsample from, with shape (N, D).
        k (int): The number of points to select.
        
    Returns:
        np.ndarray or torch.Tensor: The selected points, with shape (k, D).
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    N, D = points.shape
    selected_indices = [np.random.randint(N)]  # Start with a random point
    distances = np.full((N,), np.inf)

    for _ in range(k - 1):
        # Compute distances to the last selected point
        new_distances = np.linalg.norm(points - points[selected_indices[-1]], axis=1)
        
        # Update the minimum distances
        np.minimum(distances, new_distances, out=distances)
        
        # Select the point with the largest minimum distance
        selected_indices.append(np.argmax(distances))
    
    return points[selected_indices]

# Example usage
#X = np.random.randn(1000, 3)  # Replace this with your dataset (poses or rig controls)
#k = 100  # Number of points to select
#subsampled_X = farthest_point_sampling(X, k)
