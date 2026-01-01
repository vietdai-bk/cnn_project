from torch.utils.data import Dataset
from PIL import Image
import glob
import torchvision.transforms as T
import os

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        
        for idx in range(10):
            paths = glob.glob(os.path.join(data_dir, str(idx), '*.jpg'))
            for p in paths:
                self.samples.append((p, idx))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = T.ToTensor()(img)
        return img, label

    def __len__(self):
        return len(self.samples)