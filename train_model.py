import os

import numpy as np
import keras
import h5py

from models import PointNetClassifier
from utils import augment_points


class DataGenerator(keras.utils.PyDataset):
    """Dataset is modified between epoch."""

    def __init__(self, x_in, y_in, batch_size, augment_data=False, **kwargs):

        super().__init__(**kwargs)
        self.x, self.y = x_in, y_in
        self.data_len = len(y_in)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_len)
        np.random.shuffle(self.indices)
        self.augment_data = augment_data

    def __len__(self):
        """Number of batches per epoch."""

        return np.ceil(self.data_len/self.batch_size).astype("int")

    def __getitem__(self, index):
        """Generate one batch of data."""

        low = index * self.batch_size
        high = np.min((low + self.batch_size, self.data_len))
        batch_indices = self.indices[low:high]

        batch_x = self.x[batch_indices]
        if self.augment_data: # Slightly different data each time
            batch_x = np.array([augment_points(x) for x in batch_x])

        batch_y = self.y[batch_indices]

        return batch_x, batch_y

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""

        self.indices = np.arange(self.data_len)
        np.random.shuffle(self.indices)


class LearningRateScheduler(keras.callbacks.Callback):
    """Initial learning rate of 0.001, divided by 2 every 20 epochs."""

    def __init__(self,):
        super().__init__()

        self.initial_lr = 0.001
        self.decay_rate = 2
        self.decay_epochs = 20

    def on_train_begin(self, logs=None):

        lr = self.model.optimizer.learning_rate
        # Set initial value
        lr.assign(self.initial_lr)
        print(f"Initial Learning rate is {float(np.array(lr)):.6f}.")

    def on_epoch_begin(self, epoch, logs=None):

        lr = self.model.optimizer.learning_rate
        # Divide by decay_rate every decay_epochs
        if epoch and not epoch % self.decay_epochs:
            lr.assign(lr/2)
            print(f"Epoch {epoch}: Learning rate is {float(np.array(lr)):.6f}.")


class BatchNormalizationMomentumScheduler(keras.callbacks.Callback):
    """The decay rate for batch normalization starts with 0.5 and is gradually 
    increased to 0.99."""

    def __init__(self,):
        super().__init__()

        self.initial_momentum = 0.5
        self.final_momentum = 0.99
        self.rate = 0.005

    def on_train_begin(self, logs=None):

        # Set new value
        self.model.bn_momentum = self.initial_momentum
        print(f"Initial BatchNormalization momentum is {self.model.bn_momentum:.3f}.")

    def on_epoch_begin(self, epoch, logs=None):

        if epoch:
            # Linear growth
            new_bn_momentum = self.initial_momentum + self.rate * epoch
            # Max value
            new_bn_momentum = np.min([new_bn_momentum, self.final_momentum])
            # Set new value
            self.model.bn_momentum = new_bn_momentum
            print(f"Epoch {epoch}: BatchNormalization momentum is {self.model.bn_momentum:.3f}.")


if __name__ == "__main__":

    # About the Model
    NUM_POINTS = 2048
    NUM_CLASSES = 40
    USE_NORMALS = False
    SAVE_DIR = "./models/saved/"
    # About the training
    MAX_EPOCH = 10
    BATCH_SIZE = 32
    # About the data
    DATA_DIR = "./data/ModelNet40_preprocessed/"

    # Load dataset (pointclouds sampled from meshes)
    data_file = h5py.File(os.path.join(DATA_DIR, "ModelNet40.hdf5"))
    train_points, train_labels = data_file["train"]["points"][...], data_file["train"]["labels"][...]
    test_points, test_labels = data_file["test"]["points"][...], data_file["test"]["labels"][...]

    # Use generators for training data augmentation
    training_generator = DataGenerator(train_points, train_labels, BATCH_SIZE, augment_data=True)
    validation_generator = DataGenerator(test_points, test_labels, BATCH_SIZE, augment_data=False)

    # Build and train the model    
    model = PointNetClassifier(NUM_CLASSES)
    model.build((BATCH_SIZE, NUM_POINTS, 6 if USE_NORMALS else 3))
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"]
    )
    
    history = model.fit(
        training_generator,
        epochs=MAX_EPOCH,
        validation_data=validation_generator,
        callbacks=[
            LearningRateScheduler(),
            BatchNormalizationMomentumScheduler()
        ]
    )

    # Save the trained model
    np.save(os.path.join(SAVE_DIR, "history_train.npy"), history.history)
    model.save(os.path.join(SAVE_DIR,"PointNetClassifier.keras"))

    # Show training curves
    try:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("accuracy")
        ax1.set_title("training curves")
        ax2.set_ylabel("loss")
        for key, val in history.history.items():
            if "accuracy" in key:
                ax1.plot(val, label=key)
            elif "loss" in key:
                ax2.plot(val, label=key, linestyle="--")
        fig.legend(loc='center', fontsize="small")
        plt.tight_layout()
        fig.savefig("docs/training_curves.png")
        plt.show()

    except ModuleNotFoundError:
        pass
