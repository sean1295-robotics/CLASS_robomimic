import torch
import torch.nn.functional as F
import numpy as np

class PairwiseDistanceCDFNormalizer:
    '''
    CDF normalizer for pairwise distances with threshold support
    Maps distances [0, threshold] to CDF values [0, 1]
    '''
    
    def __init__(self, dist_matrix, threshold=None, quantile=0.01, n_quantiles=10000, device=None):
        """
        Args:
            dist_matrix: Square distance matrix (N x N)
            threshold: Maximum distance to consider. If None, uses quantile
            quantile: Quantile to use as threshold if threshold is None
            n_quantiles: Number of quantile points for CDF approximation
            device: Device to place tensors on
        """
        self.device = device if device is not None else dist_matrix.device
        
        # Extract upper triangular distances (excluding diagonal)
        triu_mask = torch.triu(torch.ones_like(dist_matrix, dtype=torch.bool), diagonal=1)
        flat_distances = dist_matrix[triu_mask].float().to(self.device)
        
        # Determine threshold
        if threshold is None:
            k = int(flat_distances.numel() * quantile)
            threshold = torch.kthvalue(flat_distances, k)[0].item()
        
        self.threshold = float(threshold)
        
        # Filter distances up to threshold
        valid_distances = flat_distances[flat_distances <= self.threshold]
        
        if len(valid_distances) == 0:
            raise ValueError("No distances found within threshold")
            
        # Sort distances for empirical CDF
        sorted_distances, _ = torch.sort(valid_distances)
        
        # Create empirical CDF
        n_points = len(sorted_distances)
        empirical_probs = torch.arange(1, n_points + 1, dtype=torch.float32, device=self.device) / n_points
        
        # Create quantile grid for interpolation
        self.n_quantiles = min(n_quantiles, n_points)
        
        if self.n_quantiles < n_points:
            # Subsample for efficiency while keeping key points
            indices = torch.linspace(0, n_points - 1, self.n_quantiles, device=self.device).long()
            self.distances = sorted_distances[indices]
            self.cdf_values = empirical_probs[indices]
        else:
            self.distances = sorted_distances
            self.cdf_values = empirical_probs
            
        # Store min/max for clamping
        self.min_distance = self.distances[0].item()
        self.max_distance = self.distances[-1].item()
        
    def __repr__(self):
        return f'PairwiseDistanceCDFNormalizer(threshold={self.threshold:.4f}, range=[{self.min_distance:.4f}, {self.max_distance:.4f}])'
    
    def _interpolate_torch(self, x, xp, fp):
        """
        PyTorch implementation of numpy.interp
        """
        # Ensure inputs are on the same device
        x = x.to(self.device)
        xp = xp.to(self.device) 
        fp = fp.to(self.device)
        
        # Find indices for interpolation
        indices = torch.searchsorted(xp, x, right=False)
        indices = torch.clamp(indices, 0, len(xp) - 1)
        
        # Handle edge cases
        below_mask = x <= xp[0]
        above_mask = x >= xp[-1]
        
        # Get surrounding points for interpolation
        indices_left = torch.clamp(indices - 1, 0, len(xp) - 2)
        indices_right = torch.clamp(indices, 1, len(xp) - 1)
        
        x_left = xp[indices_left]
        x_right = xp[indices_right]
        y_left = fp[indices_left]
        y_right = fp[indices_right]
        
        # Linear interpolation
        weights = (x - x_left) / torch.clamp(x_right - x_left, min=1e-8)
        result = y_left + weights * (y_right - y_left)
        
        # Handle edge cases
        result[below_mask] = fp[0]
        result[above_mask] = fp[-1]
        
        return result
    
    def distance_to_cdf(self, distances):
        """
        Convert distances to CDF values [0, 1]
        Distances > threshold are mapped to CDF value 1.0
        """
        distances = distances.float().to(self.device)
        
        # Clamp distances above threshold to threshold
        clamped_distances = torch.clamp(distances, 0, self.threshold)
        
        # Interpolate to get CDF values
        cdf_vals = self._interpolate_torch(clamped_distances, self.distances, self.cdf_values)
        
        # Ensure values are in [0, 1]
        cdf_vals = torch.clamp(cdf_vals, 0.0, 1.0)
        
        return cdf_vals
    
    def cdf_to_distance(self, cdf_values):
        """
        Convert CDF values [0, 1] back to distances
        CDF values are clamped to valid range
        """
        cdf_values = cdf_values.float().to(self.device)
        
        # Clamp CDF values to valid range
        cdf_values = torch.clamp(cdf_values, self.cdf_values[0], self.cdf_values[-1])
        
        # Interpolate to get distances
        distances = self._interpolate_torch(cdf_values, self.cdf_values, self.distances)
        
        return distances
    
    def to_device(self, device):
        """Move all tensors to specified device"""
        self.device = device
        self.distances = self.distances.to(device)
        self.cdf_values = self.cdf_values.to(device)
        return self