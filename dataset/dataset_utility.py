from torch.utils.data import DataLoader
import os
from typing import Tuple
import torch_dataset


def get_train_valid_datasets(
    data_root_dir: str
) -> Tuple[torch_dataset.DiffusionDataset, torch_dataset.DiffusionDataset]:
    
    train_dir = os.path.join(data_root_dir, 'train')
    valid_dir = os.path.join(data_root_dir, 'valid')

    train_dataset = torch_dataset.DiffusionDataset(train_dir)
    valid_dataset = torch_dataset.DiffusionDataset(valid_dir)

    return train_dataset, valid_dataset


def get_train_valid_dataloaders(
    data_root_dir: str, 
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset, valid_dataset = get_train_valid_datasets(data_root_dir)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, valid_loader


