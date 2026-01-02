import torch
from datasets import MyDataset
from models import SimpleModel
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--checkpoint_folder", type=str, default="checkpoints")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--SGD", action="store_true")

def split_data(dataset, test_size=0.2):
    indices = list(range(len(dataset)))
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
        random_state=42
    )

    train_data = Subset(dataset, train_idx)
    test_data  = Subset(dataset, test_idx)

    return train_data, test_data


def train(model, train_loader, loss_function, optimizer, epoch):
    total_loss = 0
    correct = 0
    total = 0

    with tqdm(train_loader, desc=f"Epoch {epoch}") as tbar:
        for img, label in tbar:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            pred = y_pred.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            tbar.set_postfix({
                "Loss": f"{total_loss:.2f}",
                "Acc": f"{correct/total:.4f}"
            })

def valid(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Validate") as tbar:
            for img, label in tbar:
                img, label = img.to(device), label.to(device)
                y_pred = model(img)
                pred = y_pred.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

    print(f"Accuracy: {correct/total:.2f}")

args = parser.parse_args()

dataset = MyDataset(args.dataset)

train_data, test_data = split_data(dataset)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size)

model = SimpleModel()
model.to(device)
if args.SGD:
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_function = nn.CrossEntropyLoss()

for epoch in range(args.epoch):
    train(model, train_loader, loss_function, optimizer, epoch+1)

torch.save(model.state_dict(),f'{args.checkpoint_folder}/SimpleModel.pt')

valid(model, test_loader)

# run: python train.py --dataset ../../mnist_dataset/trainingSet  --epoch 5 --batch_size 64 --lr 1e-4 --SGD