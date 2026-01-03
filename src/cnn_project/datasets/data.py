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

        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        for cls_name in self.classes:
            folder_path = os.path.join(data_dir, cls_name)
            label = self.class_to_idx[cls_name]

            for ext in extensions:
                for p in glob.glob(os.path.join(folder_path, ext)):
                    self.samples.append((p, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.samples)