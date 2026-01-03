from models import SimpleModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--image_path", type=str)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel()

feature_maps = {}

def hook_fn(module, input, output):
    feature_maps["block1_conv"] = output.detach().cpu()

hook_handle = model.block1.block[0].register_forward_hook(hook_fn)

def show_fm(model_path, im_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    im = Image.open(im_path).convert("L")
    im_tensor = T.ToTensor()(im)
    im_tensor = im_tensor.to(device)

    with torch.no_grad():
        out = model(im_tensor.unsqueeze(0))

    fm = feature_maps["block1_conv"].squeeze(0)
    num_maps = fm.shape[0]

    cols = 4
    rows = (num_maps + cols - 1) // cols

    plt.figure(figsize=(12, 6))
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fm[i], cmap="gray")
        plt.title(f"Map {i+1}")
        plt.axis("off")

    plt.suptitle("Feature Maps (conv1)")
    plt.tight_layout()
    plt.show()

args = parser.parse_args()
show_fm(args.model_path, args.image_path)

hook_handle.remove()