
import os
import json

import numpy as np
import keras

from utils import preprocess_mesh


if __name__ == "__main__":
    
    DATA_DIR = "./data"
    SAVE_DIR = "./models/saved/"
    NUM_POINTS = 2048

    # Load example
    points = preprocess_mesh(os.path.join(DATA_DIR, "example/desk_chair.off"), NUM_POINTS)
    batch = np.expand_dims(points, axis=0) # (2048, 3) -> (1, 2048, 3)

    # Load model & predict
    model = keras.saving.load_model(os.path.join(SAVE_DIR, "PointNetClassifier.keras"))
    preds = model.predict(batch)

    # Return result
    preds = preds.flatten() # (1, 40) -> (40, )
    label = np.argmax(preds.flatten())
    with open(os.path.join(DATA_DIR, "ModelNet40_preprocessed/class_map.json"), "r") as f:
        # "inverted" dict stored in json file
        CLASS_MAP = {v: k for k, v in json.load(f).items()}
    result = f"predicted class '{CLASS_MAP.get(label)}' with probability {preds[label]:.3f}"

    try:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=points[:, 2], cmap='twilight_shifted')
        ax.view_init(10, 45)
        ax.set_axis_off()
        ax.set_title(result)
        fig.tight_layout()
        fig.savefig("docs/pred_example.png")
        plt.show()

    except ModuleNotFoundError:
        print(result)
