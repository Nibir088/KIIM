from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
class ProjectionModule(nn.Module):
    """
    Module for combining network logits with spatial priors using learnable weights.
    
    Args:
        num_classes (int): Number of output classes
        init_value (float, optional): Initial value for weights. Defaults to 0.5
    
    Attributes:
        weights (nn.Parameter): Learnable weights for spatial priors [1, num_classes, 1, 1]
    """
    
    def __init__(
        self,
        num_classes: int = 4
    ):
        super().__init__()
        
        # Initialize learnable weights of shape [1, num_classes, 1, 1]
        # Using nn.Parameter to make weights learnable during training
        self.weights = nn.Parameter(
            torch.full((1, num_classes, 1, 1), 1.0)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        spatial_priors: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine network logits with spatial priors using learnable weights.
        
        Args:
            logits (torch.Tensor): Network output logits [B, num_classes, H, W]
            spatial_priors (torch.Tensor): Spatial priors [B, num_classes, H, W]
            
        Returns:
            torch.Tensor: Combined predictions [B, num_classes, H, W]
        """
        # Validate input shapes
        if logits.shape != spatial_priors.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} != spatial_priors {spatial_priors.shape}"
            )
        
        if logits.shape[1] != self.weights.shape[1]:
            raise ValueError(
                f"Number of classes mismatch: logits {logits.shape[1]} != weights {self.weights.shape[1]}"
            )
        
        # Combine logits with weighted spatial priors
        # weights is broadcast from [1, C, 1, 1] to [B, C, H, W]
        # ensemble = F.softmax(logits, dim=1) + self.weights * spatial_priors
        # ensemble = F.softmax(logits, dim=1) + F.softmax(self.weights * spatial_priors)
        ensemble = logits + self.weights * spatial_priors
        output = {}
        output['weighted_ensemble'] = ensemble
        output['CPM_feature'] = spatial_priors
        return output