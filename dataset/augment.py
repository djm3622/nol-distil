from torchvision import transforms

class DiffusionAugments:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def re_normalize(self, image):
        return (image - image.mean(dim=(-2, -1), keepdim=True)) / image.std(dim=(-2, -1), keepdim=True)

    def __call__(self, image):
        image = self.transform(image)
        return self.re_normalize(image)
