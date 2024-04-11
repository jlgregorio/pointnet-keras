import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, BatchNormalization, Dense, ReLU, GlobalMaxPooling1D

# TODO: use a Regulizer (Keras) in PointNetTransform Layer

class PointNetTransform(Layer):
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
        
        # num_features or dimension D (= 3 by default, for x, y, z)
        self.num_features = input_shape[-1]

        self.shared_mlp_1 = PointNetSharedMLP(64, bn_momentum=self.bn_momentum)
        self.shared_mlp_2 = PointNetSharedMLP(128, bn_momentum=self.bn_momentum)
        self.shared_mlp_3 = PointNetSharedMLP(1024, bn_momentum=self.bn_momentum)
        
        self.mlp_1 = PointNetMLP(512, bn_momentum=self.bn_momentum)
        self.mlp_2 = PointNetMLP(256, bn_momentum=self.bn_momentum)

        # Trainable weights
        self.w = self.add_weight(
            shape=(256, self.num_features**2),
            initializer=tf.zeros_initializer,
            trainable=True,
            name='w'
        )

        # Trainable biases
        self.b = self.add_weight(
            shape=(self.num_features, self.num_features),
            initializer=tf.zeros_initializer,
            trainable=True,
            name='b'
        )

        # Initialize bias with identity
        I = tf.constant(np.eye(self.num_features), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)


    def call(self, x, training=None):

        # Keep trace of input for final matrix multiply
        # (B, N, D)
        input_x = x 

        # Embed to higher dim
        # (B, N, D) -> (B, N, 1, 3/D)
        x = tf.expand_dims(input_x, axis=2)
        x = self.shared_mlp_1(x, training=training)
        x = self.shared_mlp_2(x, training=training)
        x = self.shared_mlp_3(x, training=training)
        # (B, N, 1, 3/D) -> (B, N, 1024)
        x = tf.squeeze(x, axis=2)

        # Global features
        # (B, N, 1024) -> (B, 1024)
        x = GlobalMaxPooling1D()(x)

        # Fully-connected layers
        # (B, 1024) -> (B, 512) -> (B, 256)
        x = self.mlp_1(x, training=training)
        x = self.mlp_2(x, training=training)

        # Convert to (B, B) matrix for matrix multiplication with input
        # (B, 256) -> (B, 1, 256)
        x = tf.expand_dims(x, axis=1)
        # (B, 1, 256) -> (B, 1, D*D)
        x = tf.matmul(x, self.w)
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.num_features, self.num_features))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.regularization:
            eye = tf.constant(np.eye(self.num_features), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(PointNetTransform, self).get_config()
        config.update({
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PointNetSharedMLP(Layer):
    """The PointNet shared MLP (or convolution) layer used for features encoding.
    
    It consists of a shared (convulutional) layer, a batch normalization layer 
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

        self.conv = Conv2D(
            self.filters,
            kernel_size=(1, 1),
            input_shape=batch_input_shape
        )
        self.bn = BatchNormalization(momentum=self.bn_momentum)
        self.activation = ReLU()

    def call(self, x, training=None):

        x = self.conv(x)
        x = self.bn(x, training)

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


class PointNetMLP(Layer):
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

        self.dense = Dense(self.units, input_shape=batch_input_shape)
        self.bn = BatchNormalization(momentum=self.bn_momentum)
        self.activation = ReLU()

    def call(self, x, training=None):

        x = self.dense(x)
        x = self.bn(x, training)

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
