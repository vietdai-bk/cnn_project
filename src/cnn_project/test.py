from utils import metric
from datasets import MyDataset
from torch.utils.data import DataLoader
from models import SimpleModel
import torch
import seaborn as sb
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str)
parser.add_argument("--acc", action="store_true")
parser.add_argument("--cfm", action="store_true")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = MyDataset(args.test_data)
test_loader = DataLoader(test_data, batch_size=args.batch_size)

model = SimpleModel()
model.eval()
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
model.to(device)

accuracy, cf, report = metric(model, test_loader)
print(report)
if args.acc:
    print(f"Accuracy: {accuracy:.2f}")
if args.cfm:
    sb.heatmap(cf, annot=True, fmt="d", cmap="Blues", xticklabels=range(10),
    yticklabels=range(10))
    plt.xlabel("Pred")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")
    plt.show()

# run: python test.py --test_data ../../mnist_dataset/trainingSet --model_path checkpoints/SimpleModel.pt --acc --cfm