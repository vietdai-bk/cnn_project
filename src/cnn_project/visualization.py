import matplotlib.pyplot as plt
import numpy as np
from inference import infer

def show_data(data):
    fig, axis = plt.subplots(1, 10, figsize=(10, 4))
    for i in range(10):
        j = np.random.randint(len(data))
        axis[i].imshow(data[j][0].squeeze(0), cmap='gray')
        axis[i].set_title(f'Label: {data[j][1]}')
        axis[i].grid(False)
        axis[i].axis('off')
    plt.show()