import os
import sys
import json

import numpy as np
import tensorflow as tf

# Relative import
DATA_DIR = os.path.dirname(__file__)
sys.path.append(os.path.dirname(DATA_DIR))
from utils import preprocess_mesh


def preprocess_ModelNet40_dataset(data_dir, class_map, num_points=2048, compute_normals=False):
    """Parse through the ModelNet40 data folders.
    
    Each mesh is loaded and sampled into a point cloud, which is then preprocessed.
    
    Args:
        data_dir: the directory containing the ModelNet40 dataset.
        class_map: classes to load and corresponding labels.
        num_points: number of points to sample per mesh
        compute_normals: associate a unit normal vector to each point.
    """

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    # The original ModelNet40 consists of 12,311 CAD-generated meshes 
    # in 40 categories/classes (such as airplane, car, plant, lamp), of which
    # 9,843 are used for training while the rest 2,468 are reserved for testing.
    for class_id, class_name in class_map.items():
        # The ModelNet40 dataset is organized in 40 folders (1 folder/class)
        print("processing class: {}".format(class_name))
        # Each folder has a subfolder for meshes used for training and a
        # subfolder for meshes used for testing
        train_folder = os.path.join(data_dir, class_name, "train/")
        test_folder = os.path.join(data_dir, class_name, "test/")

        # Each mesh has to be converted into a pointcloud before use in PointNet
        for f in os.listdir(train_folder):
            filepath = os.path.join(train_folder, f)
            train_points.append(preprocess_mesh(filepath, num_points, compute_normals, augment_data=True))
            train_labels.append(class_id)

        for f in os.listdir(test_folder):
            filepath = os.path.join(test_folder, f)
            test_points.append(preprocess_mesh(filepath, num_points, compute_normals))
            test_labels.append(class_id)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels)
    )


if __name__ == "__main__":

    # About the data
    MN40_DIR = "./ModelNet40/"
    NUM_POINTS = 2048
    USE_NORMALS = False
    CLASS_MAP = {
        0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf',
        5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone',
        10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 
        15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp', 
        20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand', 24: 'person', 
        25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink', 
        30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 
        35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'
        }

    # Load dataset
    train_points, test_points, train_labels, test_labels = preprocess_ModelNet40_dataset(MN40_DIR, CLASS_MAP, NUM_POINTS, USE_NORMALS)
    # Check if data has been correctly loaded
    assert train_points.shape==(9843, NUM_POINTS, 6 if USE_NORMALS else 3)
    assert train_labels.shape==(9843,)
    assert test_points.shape==(2468, NUM_POINTS, 6 if USE_NORMALS else 3)
    assert test_labels.shape==(2468,)

    # Convert to Tensorflow Dataset format
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    # Shuffle the data so the different classes are not in order (for training)
    train_dataset = train_dataset.shuffle(len(train_points))
    test_dataset = test_dataset.shuffle(len(test_points))
    # Save
    train_dataset.save(os.path.join(DATA_DIR, "ModelNet_preprocessed/ModelNet40_train"))
    test_dataset.save(os.path.join(DATA_DIR,"ModelNet_preprocessed/ModelNet40_test"))

    # Save CLASS_MAP (for later inference use)
    with open(os.path.join(DATA_DIR,'ModelNet_preprocessed/class_map.json', 'w')) as f:
            # cannot use int as key so store "inverted" dict
            json.dump(dict(map(reversed, CLASS_MAP.items())), f)
