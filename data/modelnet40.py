import os
import sys
import json

import numpy as np
import h5py

# Relative import
DATA_DIR = os.path.dirname(__file__)
sys.path.append(os.path.dirname(DATA_DIR))
from utils import preprocess_mesh


def parse_ModelNet40_dataset(data_dir, split="original", test_size=0.2):
    """Parse through the ModelNet40 data folders.

    Args:
        data_dir: the directory containing the ModelNet40 dataset.
        split: either "original" or "random".
        test_size: the proportion of the dataset to include in the test split.
    """

    train_files = []
    train_labels = []
    test_files = []
    test_labels = []
    
    # Correspondence between class id and names (i.e. class 0 is "airplane")
    class_map = {id: name for id, name in enumerate(os.listdir(data_dir))}

    # The original ModelNet40 consists of 12,311 CAD-generated meshes 
    # in 40 categories/classes (such as airplane, car, plant, lamp), of which
    # 9,843 are used for training while the rest 2,468 are reserved for testing.
    for class_id, class_name in class_map.items():
        # The ModelNet40 dataset is organized in 40 folders (1 folder/class)
        # Each folder has a subfolder for meshes used for training and a
        # subfolder for meshes used for testing
        train_folder = os.path.join(data_dir, class_name, "train/")
        test_folder = os.path.join(data_dir, class_name, "test/")
        
        train_folder_files = []
        test_folder_files = []
        
        for f in os.listdir(train_folder):
            filepath = os.path.join(train_folder, f)
            train_folder_files.append(filepath)
            
        for f in os.listdir(test_folder):
            filepath = os.path.join(test_folder, f)
            test_folder_files.append(filepath)
        
        if split=="original": # Keep the original train/test split
            
            train_files.extend(train_folder_files)
            train_labels.extend(len(train_folder_files) * [class_id])
            test_files.extend(test_folder_files)
            test_labels.extend(len(test_folder_files) * [class_id])
        
        elif split=="random": # Random train/test split
        
            # Mix files contained in train and test folders
            all_files = train_folder_files + test_folder_files
            np.random.shuffle(all_files)
            # Split them according to test_size
            n_test = int(test_size * len(all_files))
            train_files.extend(all_files[:-n_test])
            train_labels.extend((len(all_files) - n_test) * [class_id])
            test_files.extend(all_files[-n_test:])
            test_labels.extend(n_test * [class_id]) 
            

    return (
        np.array(train_files),
        np.array(train_labels),
        np.array(test_files),
        np.array(test_labels),
        class_map
    )


if __name__ == "__main__":

    MN40_DIR = "/home/localadmin/jean-loup/datasets/ModelNet40"
    NUM_POINTS = 2048
    USE_NORMALS = False

    # Get ModelNet40 dataset content
    train_files, train_labels, test_files, test_labels, class_map = parse_ModelNet40_dataset(MN40_DIR)

    # Compute pointclouds
    train_points = np.array([preprocess_mesh(f, NUM_POINTS, USE_NORMALS, augment_data=False)
                             for f in train_files], dtype="float32")
    test_points = np.array([preprocess_mesh(f, NUM_POINTS, USE_NORMALS, augment_data=False)
                            for f in test_files], dtype="float32")

    # Store result in a HDF5 file
    with h5py.File(os.path.join(DATA_DIR, "./ModelNet40_preprocessed/ModelNet40.hdf5"), "w") as file:
        
        # One group for the training data
        train = file.create_group("train")
        train.create_dataset("files", data=train_files.astype("S")) # HDF5 doesn't handle unicode strings
        train.create_dataset("labels", dtype="int32", data=train_labels)
        train.create_dataset("points", dtype="float32", data=train_points, chunks=True)

        # One group for the testing data
        test = file.create_group("test")
        test.create_dataset("files", data=test_files.astype("S"))
        test.create_dataset("labels", dtype="int32", data=test_labels)
        test.create_dataset("points", dtype="float32", data=test_points, chunks=True)

        # Metadata
        file.attrs["name"] =  np.array(["ModelNet40 dataset"], dtype="S")
        file.attrs["info"] =  np.array(["consists of 12,311 CAD-generated meshes"
                                        "in 40 categories/classes"], dtype="S")
        file.attrs["classes"] =  np.array([*class_map.values()], dtype="S")
        
    # Save class_map as a separate file for later inference use
    with open(os.path.join(DATA_DIR,'./ModelNet40_preprocessed/class_map.json'), 'w') as file:
            # cannot use int as key so store "inverted" dict
            json.dump(dict(map(reversed, class_map.items())), file)

