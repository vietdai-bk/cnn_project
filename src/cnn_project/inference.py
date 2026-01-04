from models import SimpleModel, ShuffleNet
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--image_path", type=str)
parser.add_argument("--pretrained", action="store_true")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(model, model_path, im_path, classes=None):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    im = Image.open(im_path).convert("RGB")
    im_tensor = transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(im_tensor)
        pred = out.argmax(dim=1)
    
    if classes:
        pred = classes[pred.item()]
    else:
        pred = pred.item()

    plt.imshow(im)
    plt.title(f"Pred: {pred}")
    plt.axis("off")
    plt.show()

    
args = parser.parse_args()
classes = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenorange"]
if args.pretrained:
    model = ShuffleNet(num_classes=6).to(device)
else:
    model = SimpleModel(in_c=3, num_classes=6).to(device)
    
summary(model, input_size=(3, 224, 224))
infer(model, args.model_path, args.image_path, classes)

# run: python inference.py --model_path checkpoints/ShuffleNet.pt --image_path image_test/test2.png --pretrained