import torch

class IrmDataLoader(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        x = self.input_data[index]
        y = self.target_data[index]
        return x, y
