from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(model, test_loader):
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
    return round(correct/total, 2)

def conf_matrix(model, test_loader):
    all_pred = []
    all_true = []
    with torch.no_grad():
        with tqdm(test_loader, desc="Validate") as tbar:
            for img, label in tbar:
                img, label = img.to(device), label.to(device)
                y_pred = model(img)
                pred = y_pred.argmax(dim=1)
                all_pred.extend(pred.cpu().numpy().flatten())
                all_true.extend(label.cpu().numpy().flatten())
    cf = confusion_matrix(all_true, all_pred)
    return cf