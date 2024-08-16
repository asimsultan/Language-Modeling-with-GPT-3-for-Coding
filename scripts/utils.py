
import torch
import pandas as pd
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['prompt']
        self.targets = data['completion']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(data):
    preprocessed_data = {
        "prompt": data['prompt'].tolist(),
        "completion": data['completion'].tolist()
    }
    return preprocessed_data
