import os
import shutil
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


def main():
    data_dir = "./data/tiny-imagenet-200" 
    zip_file = "./data/tiny-imagenet-200.zip"
    train_dir = "./data/train"
    valid_dir = "./data/valid"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_bar = tqdm(
        train_dataset,
        desc='Training Set',
        leave=False,
        mininterval=1.0
    )
    for i, (img, label) in enumerate(train_bar):
        save_image(img, os.path.join(train_dir, f"train_{i:04d}_{label}.png"))

    val_bar = tqdm(
        val_dataset,
        desc='Validation Set',
        leave=False,
        mininterval=1.0
    )
    for i, (img, label) in enumerate(val_bar):
        save_image(img, os.path.join(valid_dir, f"val_{i:04d}_{label}.png"))

    print(f"Deleting {data_dir} and {zip_file}...")
    shutil.rmtree(data_dir)
    if os.path.exists(zip_file):
        os.remove(zip_file)  
    print(f"{data_dir} and {zip_file} deleted.")


if __name__ == "__main__":
    main()
