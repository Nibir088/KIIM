# from typing import Optional, Dict, Any
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from pathlib import Path
# import yaml
# import json
# from data.dataset import *
# import torch

# class IrrigationDataModule(pl.LightningDataModule):
#     """
#     PyTorch Lightning DataModule for irrigation dataset.
    
#     This version assumes separate paths for train/val/test data and all necessary
#     ImageMaskDataset parameters are provided via config.
#     """
    
#     def __init__(
#         self,
#         config_path: str,
#         override_params: Optional[Dict[str, Any]] = None,
#         merge_train_valid: Optional[bool] = False
#     ):
#         """
#         Initialize the DataModule.
        
#         Args:
#             config_path: Path to YAML configuration file
#             override_params: Optional dictionary to override config parameters
#         """
#         super().__init__()
#         self.config_path = config_path
#         self.override_params = override_params or {}
        
#         # Load and update configuration
#         self.config = self._load_config()
#         self._update_config(self.override_params)
        
#         # Initialize dataset attributes
#         self.train_dataset = None
#         self.val_dataset = None
#         self.test_dataset = None
        
#         # Store parameters from config
#         self.dataset_params = self.config.get('dataset', {})
#         self.dataloader_params = self.config.get('dataloader', {})
        
        
#         self.data_file = json.load(open(self.dataset_params.get('data_file_base_name'),'r'))
#         self.data_file_paths = self.data_file[self.dataset_params.get('data_file_index')]
        
#         self.merge = merge_train_valid
        
#     def _load_config(self) -> Dict[str, Any]:
#         """Load and return configuration from YAML file."""
#         with open(self.config_path, 'r') as f:
#             return yaml.safe_load(f)
    
#     def _update_config(self, override_params: Dict[str, Any]) -> None:
#         """Update configuration with override parameters."""
#         for key, value in override_params.items():
#             keys = key.split('.')
#             current = self.config
#             for k in keys[:-1]:
#                 current = current.setdefault(k, {})
#             current[keys[-1]] = value
    
#     def setup(self, stage: Optional[str] = None) -> None:
#         """
#         Set up datasets for training, validation, and testing.
        
#         Args:
#             stage: Either 'fit', 'test', or None
#         """
        
        
#         # Create datasets based on stage
#         if stage == 'fit' or stage is None:
#             if (self.data_file_paths['train']['len']>0):
#                 self.train_dataset = ImageMaskDataset(
#                     image_paths = self.data_file_paths['train']['image'],
#                     mask_paths = self.data_file_paths['train']['label'],
#                     crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
#                     states=self.dataset_params.get('states', []),
#                     image_size=self.dataset_params.get('image_size', (256, 256)),
#                     transform=self.dataset_params.get('transform', False),
#                     gamma_value=self.dataset_params.get('gamma_value', 1.3),
#                     is_binary=self.dataset_params.get('is_binary', False),
#                     image_types=self.dataset_params.get('image_types', ['image']),
#                     agri_indices=self.dataset_params.get('agri_indices', [])
#                 )
#             if (self.data_file_paths['valid']['len']>0):
#                 self.val_dataset = ImageMaskDataset(
#                     image_paths = self.data_file_paths['valid']['image'],
#                     mask_paths = self.data_file_paths['valid']['label'],
#                     crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
#                     states=self.dataset_params.get('states', []),
#                     image_size=self.dataset_params.get('image_size', (256, 256)),
#                     transform=self.dataset_params.get('transform', False),
#                     gamma_value=self.dataset_params.get('gamma_value', 1.3),
#                     is_binary=self.dataset_params.get('is_binary', False),
#                     image_types=self.dataset_params.get('image_types', ['image']),
#                     agri_indices=self.dataset_params.get('agri_indices', [])
#                 )
                
#                 if self.merge:
#                     self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
        
#         if (stage == 'test' or stage is None) and (self.data_file['test']['len']>0):
            
#             self.test_dataset = {}
#             states = self.dataset_params.get('states', [])
#             for state in states:
#                 data_file_paths = self.data_file[state]
#                 test_data = ImageMaskDataset(
#                     image_paths = data_file_paths['test']['image'],
#                     mask_paths = data_file_paths['test']['label'],
#                     crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
#                     states=states,
#                     image_size=self.dataset_params.get('image_size', (256, 256)),
#                     transform=self.dataset_params.get('transform', False),
#                     gamma_value=self.dataset_params.get('gamma_value', 1.3),
#                     is_binary=self.dataset_params.get('is_binary', False),
#                     image_types=self.dataset_params.get('image_types', ['image']),
#                     agri_indices=self.dataset_params.get('agri_indices', [])
#                 )
#                 self.test_dataset[state] = test_data
            
                            
                
    
#     def _get_dataloader_kwargs(self) -> Dict[str, Any]:
#         """Get keyword arguments for DataLoader from config."""
#         return {
#             'batch_size': self.dataloader_params.get('batch_size', 32),
#             'num_workers': self.dataloader_params.get('num_workers', 4),
#             'pin_memory': self.dataloader_params.get('pin_memory', True),
#             'shuffle': False
#         }
    
#     def train_dataloader(self) -> DataLoader:
#         """Return the training DataLoader."""
#         kwargs = self._get_dataloader_kwargs()
#         kwargs['shuffle'] = True  # Enable shuffling for training
#         return DataLoader(self.train_dataset, **kwargs)
    
#     def val_dataloader(self) -> DataLoader:
#         """Return the validation DataLoader."""
#         return DataLoader(self.val_dataset, **self._get_dataloader_kwargs())
    
#     def test_dataloader(self) -> DataLoader:
#         """Return the test DataLoader."""
#         if not hasattr(self, 'test_dataset') or len(self.test_dataset) == 0:
#             raise ValueError("Test datasets are not initialized or are empty.")

#         dataloaders = {
#             state: DataLoader(dataset, **self._get_dataloader_kwargs())
#             for state, dataset in self.test_dataset.items()
#         }
#         return dataloaders
#         # return DataLoader(self.test_dataset, **self._get_dataloader_kwargs())



from typing import Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data.dataset import ImageMaskDataset
import json

class IrrigationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for irrigation dataset.

    This version assumes that the configuration is provided directly as a dictionary.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        merge_train_valid: Optional[bool] = False,
    ):
        """
        Initialize the DataModule.

        Args:
            config: Configuration dictionary containing dataset and dataloader parameters.
            merge_train_valid: If True, merges train and validation datasets into one.
        """
        super().__init__()
        self.config = config
        self.merge = merge_train_valid

        # Extract configuration parameters
        self.dataset_params = self.config.get('dataset', {})
        self.dataloader_params = self.config.get('dataloader', {})
        
        self.data_file = json.load(open(self.dataset_params.get('data_file_base_name'),'r'))
        self.data_file_paths = self.data_file[self.dataset_params.get('data_file_index')]


        # self.data_file = self.dataset_params.get('data_file_base_name')
        # self.data_file_paths = self.dataset_params.get('data_file_index', {})

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage: Either 'fit', 'test', or None
        """
        if stage == 'fit' or stage is None:
            if (self.data_file_paths['train']['len']>0):
                self.train_dataset = ImageMaskDataset(
                    image_paths = self.data_file_paths['train']['image'],
                    mask_paths = self.data_file_paths['train']['label'],
                    crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
                    states=self.dataset_params.get('states', []),
                    image_size=self.dataset_params.get('image_size', (256, 256)),
                    transform=self.dataset_params.get('transform', False),
                    gamma_value=self.dataset_params.get('gamma_value', 1.3),
                    is_binary=self.dataset_params.get('is_binary', False),
                    image_types=self.dataset_params.get('image_types', ['image']),
                    agri_indices=self.dataset_params.get('agri_indices', [])
                )
            if ('len' in self.data_file_paths['valid'].keys()) and (self.data_file_paths['valid']['len']>0):
                self.val_dataset = ImageMaskDataset(
                    image_paths = self.data_file_paths['valid']['image'],
                    mask_paths = self.data_file_paths['valid']['label'],
                    crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
                    states=self.dataset_params.get('states', []),
                    image_size=self.dataset_params.get('image_size', (256, 256)),
                    transform=self.dataset_params.get('transform', False),
                    gamma_value=self.dataset_params.get('gamma_value', 1.3),
                    is_binary=self.dataset_params.get('is_binary', False),
                    image_types=self.dataset_params.get('image_types', ['image']),
                    agri_indices=self.dataset_params.get('agri_indices', [])
                )
            else:
                self.val_dataset = None
                
                if self.merge:
                    self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
        
        if (stage == 'test' or stage is None):
            
            self.test_dataset = {}
            states = self.dataset_params.get('states', [])
            # states = ['FL','UT']
            for state in states:
                data_file_paths = self.data_file[state]
                test_data = ImageMaskDataset(
                    image_paths = data_file_paths['test']['image'],
                    mask_paths = data_file_paths['test']['label'],
                    crop_matrices_path=self.dataset_params.get('crop_matrices_path'),
                    states=states,
                    image_size=self.dataset_params.get('image_size', (256, 256)),
                    transform=self.dataset_params.get('transform', False),
                    gamma_value=self.dataset_params.get('gamma_value', 1.3),
                    is_binary=self.dataset_params.get('is_binary', False),
                    image_types=self.dataset_params.get('image_types', ['image']),
                    agri_indices=self.dataset_params.get('agri_indices', [])
                )
                self.test_dataset[state] = test_data

    def _get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for DataLoader from config."""
        return {
            'batch_size': self.dataloader_params.get('batch_size', 32),
            'num_workers': self.dataloader_params.get('num_workers', 4),
            'pin_memory': self.dataloader_params.get('pin_memory', True),
            'shuffle': False,
        }

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        kwargs = self._get_dataloader_kwargs()
        kwargs['shuffle'] = False  # Enable shuffling for training
        kwargs['drop_last'] = True
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs())

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Return the test DataLoaders for each state."""
        if not self.test_dataset:
            raise ValueError("Test datasets are not initialized or are empty.")

        return {
            state: DataLoader(dataset, **self._get_dataloader_kwargs())
            for state, dataset in self.test_dataset.items()
        }
