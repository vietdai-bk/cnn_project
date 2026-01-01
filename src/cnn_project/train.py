import torch
from data import MyDataset
from visualization import show_data
from model import SimpleModel
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data_dir = "../../mnist_dataset/trainingSet"
dataset = MyDataset(data_dir)

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

train_data, test_data = split_data(dataset)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

def train(model, train_loader, loss_function, optimizer, epoch):
    model.train()
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
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Validate") as tbar:
            for img, label in tbar:
                img, label = img.to(device), label.to(device)
                y_pred = model(img)
                pred = y_pred.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                all_pred.extend(pred.cpu().numpy().flatten())
                all_true.extend(label.cpu().numpy().flatten())

    print(f"Accuracy: {correct/total:.2f}")
    return all_pred, all_true

# show_data(train_data)
model = SimpleModel()
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

# for epoch in range(10):
#     train(model, train_loader, loss_function, optimizer, epoch+1)

# torch.save(model.state_dict(), 'models/SimpleModel.pt')

# Validate
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleModel()
model_path = 'models/SimpleModel.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
pred, label = valid(model, test_loader)
cf = confusion_matrix(label, pred)
sb.heatmap(cf, annot=True, fmt="d", cmap="Blues")
plt.show()