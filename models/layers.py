import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class PointNetTransform(layers.Layer):
    """The PointNet transform layer used for features encoding.
    
    It consists of a T-Net predicting an input-dependant transformation matrix 
    and a matrix multiply operation between aforementioned matrix and the input.
    """
    
    def __init__(self, regularization=False, bn_momentum=0.99, **kwargs):
        """Args:
            regularization: constrain the transformation matrix to be close to 
            orthogonal.
            bn_momentum: the momentum used for the batch normalization layer.
        """
        
        super(PointNetTransform, self).__init__(**kwargs)
        self.regularization = regularization
        self.bn_momentum = bn_momentum

    def build(self, input_shape):
        
        # num_features or dimension D
        # e.g., D=3 for the input transform net for and D=64 for the feature 
        # transform net in the PointNet original paper
        self.num_features = input_shape[-1]

        self.shared_mlp_1 = PointNetSharedMLP(64, bn_momentum=self.bn_momentum)
        self.shared_mlp_2 = PointNetSharedMLP(128, bn_momentum=self.bn_momentum)
        self.shared_mlp_3 = PointNetSharedMLP(1024, bn_momentum=self.bn_momentum)
        
        self.max_pool = layers.GlobalMaxPooling1D()

        self.mlp_1 = PointNetMLP(512, bn_momentum=self.bn_momentum)
        self.mlp_2 = PointNetMLP(256, bn_momentum=self.bn_momentum)

        reg = OrthogonalRegularizer(self.num_features) if self.regularization else None
        self.transform = layers.Dense(
            self.num_features**2,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(
                np.eye(self.num_features).flatten()
            ),
            activity_regularizer=reg
        )


    def call(self, x, training=None):

        # Keep trace of input for final matrix multiply
        input_x = x # (B, N, D)

        # Embed to higher dim
        # (B, N, D) -> (B, N, 1024)
        x = self.shared_mlp_1(x, training=training)
        x = self.shared_mlp_2(x, training=training)
        x = self.shared_mlp_3(x, training=training)

        # Global features
        # (B, N, 1024) -> (B, 1024)
        x = self.max_pool(x)

        # Fully-connected layers
        # (B, 1024) -> (B, 512) -> (B, 256)
        x = self.mlp_1(x, training=training)
        x = self.mlp_2(x, training=training)

        # Transformation net or T-net
        # (B, 256) -> (B, D, D)
        x = self.transform(x) # (B, D**2)
        x = tf.reshape(x, (-1, self.num_features, self.num_features))

        # Matrix multiply 
        # (B, D, D) * (B, D, D) = (B, N, D)
        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(PointNetTransform, self).get_config()
        config.update({
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """The regularization constrains the feature transformation matrix to be 
    close to orthogonal, and improves the optimimzation stability and the model
     performances.
    """
    
    def __init__(self, num_features, weight=1e-3):
        """Args:
            num_features: the number of features.
            weight: the weight for the regularization loss.
        """
        
        self.num_features = num_features
        self.weight = weight

    def __call__(self, x):
        
        # (B, D**2) -> (B, D, D)
        x = tf.reshape(x, (-1, self.num_features, self.num_features))   
        x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
        
        return self.weight * tf.reduce_sum((tf.eye(self.num_features) - x_xT)**2)


class PointNetSharedMLP(layers.Layer):
    """The PointNet shared MLP (or convolution) layer used for features encoding.
    
    It consists of a shared (convolutional) layer, a batch normalization layer 
    and a ReLU activation unit.
    """
    
    def __init__(self, filters, bn_momentum=0.99, **kwargs):
        """Args:
            filters: the dimensionality of the output space (number of shared layers).
            bn_momentum: the momentum used for the batch normalization layer.
        """
        
        super(PointNetSharedMLP, self).__init__(**kwargs)
        self.filters = filters
        self.bn_momentum = bn_momentum

    def build(self, batch_input_shape):

        self.conv = layers.Conv1D(
            self.filters,
            kernel_size=1,
            input_shape=batch_input_shape
        )
        self.bn = layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation = layers.ReLU()

    def call(self, x, training=None):

        x = self.conv(x)
        x = self.bn(x, training=training)

        return self.activation(x)

    def get_config(self):
        config = super(PointNetSharedMLP, self).get_config()
        config.update({
            'filters': self.filters,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PointNetMLP(layers.Layer):
    """The PointNet MLP (or dense) layer used for classification.
    
    It consists of a fully connected layer, a batch normalization layer and a
    ReLU activation unit.
    """
    
    
    def __init__(self, units, bn_momentum=0.99, **kwargs):
        """Args:
            units: the dimensionality of the output space.
            bn_momentum: the momentum used for the batch normalization layer.
        """
        
        super(PointNetMLP, self).__init__(**kwargs)
        self.units = units
        self.bn_momentum = bn_momentum

    def build(self, batch_input_shape):

        self.dense = layers.Dense(self.units, input_shape=batch_input_shape)
        self.bn = layers.BatchNormalization(momentum=self.bn_momentum)
        self.activation = layers.ReLU()

    def call(self, x, training=None):

        x = self.dense(x)
        x = self.bn(x, training=training)

        return self.activation(x)
    
    def get_config(self):
        config = super(PointNetMLP, self).get_config()
        config.update({
            'units': self.units,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
