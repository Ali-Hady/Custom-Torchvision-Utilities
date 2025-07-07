import torch

def pil_collate(batch):
    """
    Custom collate function for PIL images and labels (to be used with DataLoader).
    """
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)
