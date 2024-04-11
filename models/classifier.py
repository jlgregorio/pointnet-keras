import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalMaxPooling1D

from models import PointNetTransform, PointNetSharedMLP, PointNetMLP


class PointNetClassifier(Model):
    """The PointNet model for classification."""

    def __init__(self,
                 num_classes: int = 40,
                 bn_momentum: float = 0.99,
                 **kwargs):
        """Args:
            num_classes: the number of classes to classify.
            bn_momentum: the momentum used for the batch normalization layer.
        """
        
        super().__init__(**kwargs)
        self.num_classes=num_classes
        self.bn_momentum = tf.Variable(bn_momentum, trainable=False) # for Callback use

        # 1st transformation network (without regularization)
        self.input_transform = PointNetTransform(
            bn_momentum=self.bn_momentum,
            regularization=False,
            name="input_transform"
        )
        # 1st layers of shared MLPs
        self.shared_mlp_1 = PointNetSharedMLP(
            64,
            bn_momentum=self.bn_momentum,
            name="shared_mlp_1_64"
        )
        self.shared_mlp_2 = PointNetSharedMLP(
            64,
            bn_momentum=self.bn_momentum,
            name="shared_mlp_2_64"
        )
        # 2d transformation network (with regularization)
        self.feature_transform = PointNetTransform(
            bn_momentum=self.bn_momentum,
            regularization=True,
            name="feature_transform"
        )
        # 2d layer of shared MLPs
        self.shared_mlp_3 = PointNetSharedMLP(
            64,
            bn_momentum=self.bn_momentum,
            name="shared_mlp_3_64"
        )
        self.shared_mlp_4 = PointNetSharedMLP(
            128,
            bn_momentum=self.bn_momentum,
            name="shared_mlp_4_128"
        )
        self.shared_mlp_5 = PointNetSharedMLP(
            1024,
            bn_momentum=self.bn_momentum,
            name="shared_mlp_5_1024"
        )
        # Global feature
        self.mlp_1 = PointNetMLP(
            512,
            bn_momentum=self.bn_momentum,
            name="mlp_1_512"
        )
        self.dropout_1 = Dropout(rate=0.3)

        self.mlp_2 = PointNetMLP(
            256,
            bn_momentum=self.bn_momentum,
            name="mlp_2_256"
        )
        self.dropout_2 = Dropout(rate=0.3)
        
        # Outputs
        # Note: softmax activation is not used in the original implementation
        self.output_scores = Dense(
            self.num_classes,
            activation="softmax",
            name="output_scores"
        )


    def build(self, batch_input_shape):

        super().build(batch_input_shape)


    def call(self, input_points, training=None):
                
        # Input transformer 
        # (B, N, 3) -> (B, N, 3)
        x = self.input_transform(input_points, training)
        x = tf.expand_dims(x, axis=2) # for weight-sharing of conv
        
        # Embed to 64-dim space 
        # (B, N, 3) -> (B, N, 64) -> (B, N, 64)
        x = self.shared_mlp_1(x, training)
        x = self.shared_mlp_2(x, training)
        x = tf.squeeze(x, axis=2)

        # Feature transformer 
        # (B, N, 64) -> (B, N, 64)
        x = self.feature_transform(x, training)

        # Embed to 1024-dim space 
        # (B, N, 64) -> (B, N, 64) -> (B, N, 128) -> (B, N, 1024)
        x = tf.expand_dims(x, axis=2)
        x = self.shared_mlp_3(x, training)
        x = self.shared_mlp_4(x, training)
        x = self.shared_mlp_5(x, training)
        x = tf.squeeze(x, axis=2)

        # Global feature vector
        # (B, N, 1024) -> (B, 1024)
        x = GlobalMaxPooling1D()(x)

        # FC layers to output k scores
        # (B, 1024) -> (B, 512) -> (B, 126) -> (B, 40)
        x = self.mlp_1(x, training)
        x = self.dropout_1(x, training)
        x = self.mlp_2(x, training)
        x = self.dropout_2(x, training)

        return self.output_scores(x)
