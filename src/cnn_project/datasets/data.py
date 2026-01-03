from torch.utils.data import Dataset
from PIL import Image
import glob
import torchvision.transforms as T
import os

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        extensions = ('*.jpg', '*.png', '*.jpeg')

        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)

            if not os.path.isdir(folder_path):
                continue

            label = int(folder_name)

            for ext in extensions:
                for p in glob.glob(os.path.join(folder_path, ext)):
                    self.samples.append((p, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = T.ToTensor()(img)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)