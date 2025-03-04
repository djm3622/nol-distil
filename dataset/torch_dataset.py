import torch
from torch.utils.data import Dataset
from torchvision import datasets


class DiffusionDataset(Dataset):

    def load_dataset(self, dataset_dir):
        return datasets.ImageFolder(root=dataset_dir)
    
    # get beta, alpha, and alpha_bar from timesteps and method specifed in config 'quadradic', 'linear', 'cosine'
    
    def get_beta_alpha_alpha_bar(self, timesteps, cfg):
        pass

    def __init__(self, dataset_dir, timesteps, cfg, augmentations=None):
        self.augmentations = augmentations
        self.timesteps = timesteps
        self.dataset = self.load_dataset(dataset_dir)

        self.beta, self.alpha, self.alpha_bar = self.get_beta_alpha_alpha_bar(timesteps, cfg)

    def __len__(self):
        return len(self.dataset)
        
    def get_noise(self, sample):
        return torch.randn_like(sample)
    
    def get_random_timesteps(self):
        return torch.randint(0, self.timesteps, (1,))

    # Todo : incorporate beta, alpha, and alpha_bar

    def get_beta(self, timestep):
        pass

    def get_alpha(self, timestep):
        pass

    def get_alpha_bar(self, timestep):
        pass
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.augmentations:
            sample = self.augmentations(sample)

        noise = self.get_noise(sample)
        timestep = self.get_random_timesteps()

        return sample, noise, timestep