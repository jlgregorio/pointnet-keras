
import os
import json

import numpy as np
from matplotlib import pyplot as plt
import keras

from utils import preprocess_mesh


if __name__ == "__main__":
    
    DATA_DIR = "./data"
    SAVE_DIR = "./models/saved/"
    NUM_POINTS = 2048

    # Load example
    points = preprocess_mesh(os.path.join(DATA_DIR, "example/desk_chair.off"), NUM_POINTS)
    points = np.expand_dims(points, axis=0) # (N, 3) -> (1, N, 3)
    # Show points
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()

    # Load model & predict
    model = keras.saving.load_model(os.path.join(SAVE_DIR, "PointNetClassifier.keras"))
    preds = model.predict(points)

    # Load class map
    with open(os.path.join(DATA_DIR, "ModelNet40_preprocessed/class_map.json"), "r") as f:
        # "inverted" dict stored in json file
        CLASS_MAP = {v: k for k, v in json.load(f).items()}

    # Return result
    preds = preds.flatten()
    label = np.argmax(preds)
    print(f"predicted class: {CLASS_MAP.get(label)} with probability: {preds[label]:.3f}")
