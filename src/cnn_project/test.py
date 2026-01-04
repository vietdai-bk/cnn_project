from utils import metric
from datasets import MyDataset
from torch.utils.data import DataLoader
from models import SimpleModel, ShuffleNet
import torch
import seaborn as sb
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str)
parser.add_argument("--acc", action="store_true")
parser.add_argument("--cfm", action="store_true")
parser.add_argument("--pretrained", action="store_true")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.pretrained:
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

else:
    transform = None

test_data = MyDataset(args.test_data, transform=transform)
in_c = test_data[0][0].shape[0]
test_loader = DataLoader(test_data, batch_size=args.batch_size)

if args.pretrained:
    model = ShuffleNet(in_c)
else:
    model = SimpleModel(in_c)
model.eval()
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
model.to(device)

accuracy, cf, report = metric(model, test_loader)
print(report)
if args.acc:
    print(f"Accuracy: {accuracy:.2f}")
if args.cfm:
    sb.heatmap(cf, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Pred")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")
    plt.show()

# run: python test.py --test_data ../../mnist_dataset/trainingSet --model_path checkpoints/SimpleModel.pt --acc --cfm