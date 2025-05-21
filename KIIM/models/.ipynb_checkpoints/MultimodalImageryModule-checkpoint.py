from typing import Optional, Dict
import torch
import torch.nn as nn

class MIM(nn.Module):
    """
    Simple Multi-Input Module for concatenating RGB and agricultural index features.
    
    Args:
        use_rgb (bool): Whether to use RGB features
        use_ndvi (bool): Whether to use NDVI index
        use_ndwi (bool): Whether to use NDWI index
        use_ndti (bool): Whether to use NDTI index
    """
    
    def __init__(
        self,
        use_rgb: bool = True,
        use_ndvi: bool = True,
        use_ndwi: bool = True,
        use_ndti: bool = True
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_ndvi = use_ndvi
        self.use_ndwi = use_ndwi
        self.use_ndti = use_ndti
        
        # Calculate total number of channels
        self.total_channels = (3 if use_rgb else 0) + \
                            (1 if use_ndvi else 0) + \
                            (1 if use_ndwi else 0) + \
                            (1 if use_ndti else 0)
        
        if self.total_channels == 0:
            raise ValueError("At least one input feature must be enabled")
            
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to concatenate selected features.
        
        Args:
            data (Dict[str, torch.Tensor]): Dictionary containing:
                - 'image': RGB image tensor [B, 3, H, W]
                - 'ndvi': NDVI tensor [B, H, W]
                - 'ndwi': NDWI tensor [B, H, W]
                - 'ndti': NDTI tensor [B, H, W]
        
        Returns:
            torch.Tensor: Concatenated features [B, C, H, W]
        """
        features = []
        
        # Add RGB features if enabled
        if self.use_rgb:
            if 'image' not in data:
                raise ValueError("RGB features enabled but 'image' not found in input data")
            features.append(data['image'])
        
        # Add NDVI if enabled
        if self.use_ndvi:
            if 'ndvi' not in data:
                raise ValueError("NDVI enabled but not found in input data")
            features.append(data['ndvi'].unsqueeze(1))
            
        # Add NDWI if enabled
        if self.use_ndwi:
            if 'ndwi' not in data:
                raise ValueError("NDWI enabled but not found in input data")
            features.append(data['ndwi'].unsqueeze(1))
            
        # Add NDTI if enabled
        if self.use_ndti:
            if 'ndti' not in data:
                raise ValueError("NDTI enabled but not found in input data")
            features.append(data['ndti'].unsqueeze(1))
        
        # Concatenate all features along channel dimension
        return torch.cat(features, dim=1)
    