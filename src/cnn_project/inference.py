from model import SimpleModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel()

def infer(model_path, im_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    im = Image.open(im_path).convert("L")
    im_tensor = T.ToTensor()(im)
    im_tensor = im_tensor.to(device)
    out = model(im_tensor.unsqueeze(0))
    pred = out.argmax(dim=1)
    plt.imshow(im, cmap='gray')
    plt.title(f"Pred: {pred.item()}")
    plt.axis(False)
    plt.show()