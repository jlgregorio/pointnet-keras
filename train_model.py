import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import PointNetClassifier


class LearningRateScheduler(keras.callbacks.Callback):
    """Initial learning rate of 0.001, divided by 2 every 20 epochs."""

    def __init__(self,):
        super().__init__()

        self.initial_lr = 0.001
        self.decay_rate = 2
        self.decay_epochs = 20

    def on_epoch_begin(self, epoch, logs=None):
        
        lr = self.model.optimizer.learning_rate
        # Set initial value
        if not epoch:
            lr.assign(self.initial_lr)
        # Divide by decay_rate every decay_epochs
        elif not epoch % self.decay_epochs:
            lr.assign(lr/2)
        print(f"Epoch {epoch}: Learning rate is {float(np.array(lr)):.6f}.")


class BatchNormalizationMomentumScheduler(tf.keras.callbacks.Callback):
    """The decay rate for batch normalization starts with 0.5 and is gradually 
    increased to 0.99."""
    
    def __init__(self,):
        super().__init__()
        
        self.initial_momentum = 0.5
        self.final_momentum = 0.99
        self.rate = 0.005
    
    def on_epoch_begin(self, epoch, logs=None):
        
        # Linear growth
        new_bn_momentum = self.initial_momentum + self.rate * epoch
        # Max value
        new_bn_momentum = np.min([new_bn_momentum, self.final_momentum])
        # Set new value
        self.model.bn_momentum.assign(new_bn_momentum)
        print(f"Epoch {epoch}: BatchNormalization momentum is {float(np.array(new_bn_momentum)):.3f}.")


if __name__ == "__main__":

    # About the Model
    NUM_POINTS = 2048
    NUM_CLASSES = 40
    USE_NORMALS = False
    SAVE_DIR = "./models/saved/"
    # About the training
    MAX_EPOCH=200
    BATCH_SIZE = 32
    # About the data
    DATA_DIR = "./data/ModelNet40_preprocessed/"

    # Load dataset
    train_dataset = tf.data.Dataset.load(os.path.join(DATA_DIR, "ModelNet40_train"))
    test_dataset = tf.data.Dataset.load(os.path.join(DATA_DIR, "ModelNet40_test"))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Build and train the model    
    model = PointNetClassifier(NUM_CLASSES)
    model.build((BATCH_SIZE, NUM_POINTS, 6 if USE_NORMALS else 3))
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["sparse_categorical_accuracy"],
    )
    
    history = model.fit(
        train_dataset,
        epochs=MAX_EPOCH,
        validation_data=test_dataset,
        callbacks=[
            LearningRateScheduler(),
            BatchNormalizationMomentumScheduler()
        ]
    )

    # Save trained model
    np.save(os.path.join(SAVE_DIR, "history_train.npy"), history.history)
    model.save(os.path.join(SAVE_DIR,"PointNetClassifier.keras"))
