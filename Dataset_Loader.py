from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset:
    
    def __init__(self, path: str, img_size: int, batch_size: int, shuffle: bool):
        """
        A class to load an image dataset using torchvision.datasets.ImageFolder and provide a DataLoader for easy batch processing.

        Attributes:
            path (str): Path to the dataset directory.
            img_size (int): Image dimensions (assumed to be square).
            batch_size (int): Number of images per batch.
            shuffle (bool): Whether to shuffle the dataset.
            transform (torchvision.transforms.Compose): Image transformations.
            dataset (ImageFolder): The dataset object.
            dataloader (DataLoader): The DataLoader for batch processing.
        """
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset with the safe loader
        self.dataset = datasets.ImageFolder(root=self.path, transform=self.transforms, loader=self.safe_loader)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def safe_loader(self, path):
        """
        This function handles corrupted image files and skips them if necessary.
        """
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted image: {path}, Error: {e}")
            return None  # Returning None will ensure ImageFolder skips it

    def get_dataloader(self):
        return self.dataloader

    def get_labels(self):
        return self.dataset.class_to_idx
