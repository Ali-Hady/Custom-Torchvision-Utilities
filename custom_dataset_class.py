import pathlib
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Dict


class CustomImageDataset(Dataset):
    """Custom dataset for loading images from a directory.
    
    This is compatible with PyTorch's DataLoader and can be used for training or validation.
    
    It assumes that the images are organized in subdirectories, where each subdirectory name corresponds to a class label.
    
    It supports a random image display method and allows loading images without transformations.
    """
    def __init__(self, image_dir: pathlib.Path, transform: transforms.Compose = transforms.Compose([transforms.ToTensor()])):
        """Custom dataset for loading images from a directory.

        Args:
            image_dir (pathlib.Path): Path to the directory containing images.
            transform (transforms.Compose, optional): Transformations to apply to the images. Defaults to a basic tensor transformation.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.class_names = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
        extensions = ['*.jpg', '*.jpeg', '*.png']
        self.image_paths = [p for ext in extensions for p in image_dir.glob(f"*/{ext}")]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = [(str(p), self.class_to_idx[p.parent.name]) for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.class_to_idx[img_path.parent.name]

        image_tensor = self.transform(image)
        image.close()  # Close the image file to free resources
        
        return image_tensor, label
    
    def load_image(self, idx: int) -> Image.Image:
        """Load an image from the dataset without transformations."""
        img_path = self.image_paths[idx]
        return Image.open(img_path)
    
    def display_random_images(self, n: int = 3, seed: int = 0):
        """Display random images from the dataset.
        
        Args:
            n (int): Number of random images to display. Defaults to 3.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            
        This method displays a grid of random images along with their original sizes and class labels.
        
        maximum number of images is limited to 15 images for display purposes.
        """
        if seed is not None:
            random.seed(seed)
        
        plt.figure(figsize=(12, 6))
        
        n = min(n, 15)  # Limit to a maximum of 15 images
        for i in range(n):
            idx = random.randint(0, len(self) - 1)
            img, label = self[idx]
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(img.permute(1, 2, 0))
            ax[0].set_title(f"Image {i + 1}\nShape: {img.shape}")
            ax[0].axis(False)
            original_img = self.load_image(idx)
            ax[1].imshow(original_img)
            ax[1].set_title(f"Original Image {i + 1}\nSize: {original_img.size}")
            ax[1].axis(False)
            fig.suptitle(f"Class: {self.class_names[label]}", fontsize=16)
            original_img.close()
    
