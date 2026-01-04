import torch
import time
from models import SimpleModel, ShuffleNet
from torchsummary import summary

def measure_latency(
    model,
    input_size,
    device="cpu",
    warmup=20,
    runs=100
):
    model.eval()
    model.to(device)

    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    return (end - start) / runs * 1000


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    models = {
        "SimpleCNN": SimpleModel(in_c=3, num_classes=10),
        "ShuffleNet": ShuffleNet(num_classes=10),
    }

    input_sizes = [224]

    for name, model in models.items():
        model.to(device)
        summary(model, input_size=(3, 224, 224))
        print(f"\nModel: {name}")
        for s in input_sizes:
            latency = measure_latency(
                model,
                input_size=(1, 3, s, s),
                device=device
            )
            print(f"Input {s}x{s}: {latency:.2f} ms")
