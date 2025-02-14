import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class DiffusionDataset(Dataset):

    def load_dataset(self, dataset_dir):
        return datasets.ImageFolder(root=dataset_dir)

    def __init__(self, dataset_dir, timesteps, augmentations=None):
        self.augmentations = augmentations
        self.timesteps = timesteps
        self.dataset = self.load_dataset(dataset_dir)

    def __len__(self):
        return len(self.dataset)
        
    def get_noise(self, sample):
        return torch.randn_like(sample)
    
    def get_random_timesteps(self):
        return torch.randint(0, self.timesteps, (1,))
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.augmentations:
            sample = self.augmentations(sample)

        noise = self.get_noise(sample)
        timestep = self.get_random_timesteps()

        return sample, noise, timestep