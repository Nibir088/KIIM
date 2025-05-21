import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageProcessor:
    """Handles image loading and processing operations."""
    
    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction to the image."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return table[image]

    @staticmethod
    def load_image(path: str, 
                  use_case: str = 'image', 
                  normalize: float = 65535.0,
                  gamma: bool = False,
                  gamma_value: float = 1.3) -> np.ndarray:
        """Load and process image based on use case."""
        # Band mapping for different use cases
        band_mapping = {
            'image': [4, 3, 2],
            'agriculture': [6, 5, 2],
            'vegetation': [6, 5, 4],
            'land-water': [5, 6, 4],
            'color-infrared': [5, 4, 3],
            'urban': [7, 6, 4],
            'all-six': [2, 3, 4, 5, 6, 7]
        }
        
        bands = rasterio.open(path)
        stacked_img = []
        for band in band_mapping[use_case]:
            stacked_img.append(bands.read(band))
        
        img = np.stack(stacked_img, axis=-1)
        
        if gamma:
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            img = ImageProcessor.adjust_gamma(img, gamma=gamma_value)
        else:
            img = img / normalize
            
        return img

    @staticmethod
    def load_mask(path: str) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
        """Load mask data from file."""
        img = rasterio.open(path)
        return (torch.from_numpy(img.read(1)).long(),
                img.read(4),
                torch.from_numpy(img.read(2)).long(),
                torch.from_numpy(img.read(3)).long())

    @staticmethod
    def load_agri_index(path: str, index_names: List[str] = ['ndvi']) -> List[np.ndarray]:
        """Calculate agricultural indices from image bands."""
        bands = rasterio.open(path)
        stk_img = {f'band_{b}': np.array(bands.read(b)).astype(np.float32) 
                  for b in range(1, 8)}
        
        index_calculations = {
            'ndti': lambda x: (x['band_6'] - x['band_7']) / (x['band_6'] + x['band_7'] + 1e-10),
            'ndvi': lambda x: (x['band_5'] - x['band_4']) / (x['band_5'] + x['band_4'] + 1e-10),
            'ndwi': lambda x: (x['band_3'] - x['band_6']) / (x['band_3'] + x['band_6'] + 1e-10)
        }
        
        images = []
        for idx in index_names:
            if idx not in index_calculations:
                raise ValueError(f"Unknown index: {idx}")
                
            index = index_calculations[idx](stk_img)
            index_min, index_max = np.min(index), np.max(index)
            index_normalized = (index - index_min) / (index_max - index_min + 1e-10)
            images.append(index_normalized)
            
        return images

class ImageMaskDataset(Dataset):
    """Dataset for handling image and mask pairs with various processing options."""
    
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 crop_matrices_path: str,
                 states: List[str],
                 image_size: Tuple[int, int] = (256, 256),
                 transform: bool = False,
                 gamma_value: float = 1.3,
                 is_binary: bool = False,
                 image_types: List[str] = ['image'],
                 agri_indices: List[str] = []):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to source images
            mask_paths: List of paths to mask images
            crop_matrices_path: Base path to crop matrices
            states: List of state codes (e.g., ['AZ', 'UT'])
            image_size: Tuple of (height, width) for resizing
            transform: Whether to apply transformations
            gamma_value: Gamma correction value
            is_binary: Whether to convert masks to binary
            image_types: List of image types to process
            agri_indices: List of agricultural indices to calculate
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.gamma_value = gamma_value
        self.is_binary = is_binary
        self.image_types = image_types
        self.agri_indices = agri_indices
        
        # Load crop matrices for each state
        self.crop_matrices = {
            state: np.load(f'{crop_matrices_path}/crop_to_irrigation_matrix_{state}.npy')
            for state in states
        }
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if transform:
            self.transform.transforms.append(
                transforms.Resize(image_size)
            )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _get_state_from_path(self, path: str) -> str:
        """Extract state code from file path."""
        for state in self.crop_matrices.keys():
            if state in path:
                return state
        raise ValueError(f"No matching state found in path: {path}")
    
    def _create_crop_matrix(self, 
                          crop_info: torch.Tensor, 
                          crop_mat: np.ndarray) -> torch.Tensor:
        """Create crop matrix from crop information."""
        one_hot = torch.nn.functional.one_hot(crop_info, num_classes=21).float()
        output = torch.matmul(one_hot, 
                            torch.tensor(crop_mat).float()).permute(2, 0, 1).float()
        return output
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample."""
        data = {}
        
        # Load different image types
        for image_type in self.image_types:
            image = ImageProcessor.load_image(
                self.image_paths[idx],
                use_case=image_type,
                gamma=True,
                gamma_value=self.gamma_value
            )
            data[image_type] = self.transform(image)
        
        # Calculate agricultural indices
        if self.agri_indices:
            agri_data = ImageProcessor.load_agri_index(
                self.image_paths[idx],
                self.agri_indices
            )
            for i, index_name in enumerate(self.agri_indices):
                data[index_name] = agri_data[i]
        
        # Load mask data
        mask, qa_img, land_msk, crop_msk = ImageProcessor.load_mask(self.mask_paths[idx])
        
        # Process mask
        mask[mask == 4] = 0
        if isinstance(self.transform.transforms[0], transforms.Resize):
            mask = self.transform(mask)
        if self.is_binary:
            mask = (mask > 0).long()
        
        # Add mask data to output
        data.update({
            'true_mask': mask,
            'qa_image': qa_img,
            'land_mask': land_msk,
            'crop_mask': self._create_crop_matrix(
                crop_msk,
                self.crop_matrices[self._get_state_from_path(self.mask_paths[idx])]
            )
        })
        
        return data
