import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, sequence_length=24):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        # Slice rolling sequence window
        x = self.data[idx : idx + self.sequence_length]
        # Target is the value immediately following the sequence
        y = self.target[idx + self.sequence_length]
        return x, y

def prepare_loaders(data, target, sequence_length=24, batch_size=32, train_split=0.8):
    # Split training and validation subsets
    split_idx = int(len(data) * train_split)
    
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_target, val_target = target[:split_idx], target[split_idx:]
    
    train_dataset = TimeSeriesDataset(train_data, train_target, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, val_target, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
