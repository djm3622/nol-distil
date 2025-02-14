from torch.utils.data import DataLoader
import os
from typing import Tuple
import torch_dataset
import augment


def get_train_valid_datasets(
    data_root_dir: str,
    cfg: dict,
    data_augmentation: bool
) -> Tuple[torch_dataset.DiffusionDataset, torch_dataset.DiffusionDataset]:
    
    train_dir = os.path.join(data_root_dir, 'train')
    valid_dir = os.path.join(data_root_dir, 'valid')

    augmentation_pipeline = None
    if data_augmentation:
        augmentation_pipeline = augment.DiffusionAugments()

    timestep = cfg['unet_model']['timesteps']
    train_dataset = torch_dataset.DiffusionDataset(train_dir, timestep, augmentation_pipeline)
    valid_dataset = torch_dataset.DiffusionDataset(valid_dir, timestep)

    return train_dataset, valid_dataset


def get_train_valid_dataloaders(
    data_root_dir: str, 
    batch_size: int,
    cfg: dict
) -> Tuple[DataLoader, DataLoader]:
    
    augment = cfg['dataset']['augment']
    train_dataset, valid_dataset = get_train_valid_datasets(data_root_dir, cfg, augment)

    num_workers = cfg['dataset']['workers_per_gpu']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    return train_loader, valid_loader


