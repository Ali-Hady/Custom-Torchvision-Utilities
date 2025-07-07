from torch.utils.data import Dataset

class OffsetDataset(Dataset):
    """Dataset wrapper that applies a label offset (useful for concatenated datasets).

    Args:
        dataset (Dataset): The original dataset.
        label_offset (int): The offset to apply to the labels.
    """
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.offset = label_offset

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y + self.offset

    def __len__(self):
        return len(self.dataset)