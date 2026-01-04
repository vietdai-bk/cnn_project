import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from models import SimpleModel, ShuffleNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 6

model_path = "checkpoints/ShuffleNet.pt"

CLASSES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenorange"
]

def load_model(model_path):
    model = ShuffleNet(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

def predict(image):
    if image is None:
        return "No image"

    model = load_model(model_path)

    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    result = {
        CLASSES[i]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

    return result

with gr.Blocks(title="Fruit Classification") as demo:
    gr.Markdown("## Fruit Classification (SimpleCNN / ShuffleNet)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload image")
            btn = gr.Button("Predict")

        with gr.Column():
            output = gr.Label(num_top_classes=6, label="Prediction")

    btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=output
    )

demo.launch()