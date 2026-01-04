import torch
from datasets import MyDataset
from models import SimpleModel, ShuffleNet
from utils import show_data
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--checkpoint_folder", type=str, default="checkpoints")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--agm", action="store_true")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--pretrained", action="store_true")

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

cost_loss = []
cost_acc = []

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
        cost_acc.append(correct/total)
        cost_loss.append(total_loss)
    return cost_loss, cost_acc

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

if args.agm or args.pretrained:
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
else:
    transform = None

dataset = MyDataset(args.dataset, transform=transform)

train_data, test_data = split_data(dataset)
in_c = train_data[0][0].shape[0]
# show_data(train_data)
# show_data(test_data)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size)

if args.pretrained:
    model = ShuffleNet(in_c=in_c, num_classes=args.num_classes)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.fc.parameters():
        p.requires_grad = True
else:
    model = SimpleModel(in_c=in_c, num_classes=args.num_classes)

model.to(device)

summary(model, input_size=(in_c, 224, 224))

if args.pretrained:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_function = nn.CrossEntropyLoss()

all_cost_loss = []
all_cost_acc = []
for epoch in range(args.epoch):
    cost_loss, cost_acc = train(model, train_loader, loss_function, optimizer, epoch+1)

all_cost_loss.extend(cost_loss)
all_cost_acc.extend(cost_acc)
cost_loss = []
cost_acc = []

if args.pretrained:
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr * 1e-1
    )
    
    for epoch in range(10):
        cost_loss, cost_acc = train(model, train_loader, loss_function, optimizer, epoch+1)
    all_cost_loss.extend(cost_loss)
    all_cost_acc.extend(cost_acc)
    torch.save(model.state_dict(),f'{args.checkpoint_folder}/ShuffleNet.pt')
else:
    torch.save(model.state_dict(),f'{args.checkpoint_folder}/SimpleModel.pt')

# plt.plot(all_cost_loss)
# plt.title("Loss Map")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.savefig('loss_map_with_petrained.png')
# plt.show()
# plt.plot(all_cost_acc)
# plt.title("Accuracy Map")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.savefig('acc_map_with_pretrained.png')
# plt.show()

valid(model, test_loader)

# run: python train.py --dataset dataset/train  --epoch 10 --batch_size 64 --lr 1e-4