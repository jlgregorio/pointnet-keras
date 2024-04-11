# PointNet implementation using Keras

This is a (yet another) custom implementation of the well-known PointNet model for pointcloud classification. This was coded with the objective to keep it as simple as possible, limiting the use of third-party libraries and making the most of the Keras API (Tensorflow background).

## Getting started

### Dependencies

The code runs with `python` 3.11, `numpy` 1.26, `tensorflow` 2.14 (`keras` 2.14).

Tensorflow 2 is preferably installed using pip with

    pip install --upgrade pip
    pip install tensorflow

but may also be installed via conda with

    conda install tensorflow-gpu -c conda-forge

(not the lasted version but it works here).

### Training

PointNet is often trained on the ModelNet40 dataset, which is also chosen here. An already trained model is included in `models/saved` folder (see below for usage). The `train_model.py` file may be re-used for futher experimentations (see below for data preparation).

In detail, the use of a custom training loop has been avoided by relying on Callbacks (`tf.keras.callbacks.Callback`) to update the *learning rate* and *batch normalization momentum* during training (in accordance with the original paper/implementation).

### Predicting

The `predict_example.py` file provides a minimal example for prediction.

In detail, using the Keras API allows to save and load models (configuration, weights, etc.) quite easily. 

### Data preparation

Data often have to be preprocessed before training, which is the case for the ModelNet40 dataset. The `ModelNet40_prep.py` in the `data` folder contains all the preprocessing steps carried here. Just download the ModelNet40 dataset (put it in `data/ModelNet40` by default) and run the file. This will create a Tensorflow dataset (`tf.data.Dataset`) and save it in `data/ModelNet40_preprocessed`.

In detail, a pointcloud (consisting of *n* points, optionally with normals) is sampled from the faces of each mesh (according to face area) contained in the ModelNet40 dataset. Pointclouds are normalized into a unit sphere. An augmentation step is performed for pointclouds used for training (randomly rotating along the up axis, jittering the position of each point by gaussian noise and shuffling the points).

## About

### PointNet

The PointNet classification network (the upper part of the figure shown below) takes a pointcloud of *n* points as input and outputs a classification score for *m* classes. 

<div align="center">
  <p><img src="docs/pointnet.jpg"></p>
  <p>PointNet architecture</p>
</div>

See the [PointNet project page](https://stanford.edu/~rqi/pointnet/) for more details about Pointnet.

## ModelNet40

ModelNet40 consists of 12,311 CAD-generated meshes distributed over 40 categories/classes (of "common" objects such as airplane, car, plant, lamp, etc.). A large portion of these meshes (9,843 to be exact) are used for training and a smaller portion (2,468 to be exact) are reserved for testing.

See the [ModelNet project page](https://modelnet.cs.princeton.edu/) for more details about ModelNet40 and access to the data.

## Aknowledgement

This work was greatly inspired by previous PointNet implementations, such as:
* [The orginal one](https://github.com/charlesq34/pointnet) by Charles R. Qi et al. (using Python 2.7 & TensorFlow 1.0)
* [This one](https://github.com/luis-gonzales/pointnet_own) by Luis R. Gonzales (using Python 3.x & TensorFlow 2.0)
* [This one](https://www.tensorflow.org/graphics/api_docs/python/tfg/nn/layer/pointnet) (partial) found in the TensorFlow Graphics module

In addition to the Tensorflow/Keras official online documentation, the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd Edition)* by Aurélien Géron was of a great help and another major source of inspiration.
