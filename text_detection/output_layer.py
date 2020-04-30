# this is the output layer according to the paper
# we wil extract the 32 feature layer into three layers
import tensorflow as tf
from text_detection import feature_merging
import numpy as np

PI = np.pi


# extract the H_out layer from feature merging layer
# and pull out 1*1 conv for F_score

class feature_out(tf.keras.layers.Layer):

    def __init__(self, fec):
        super(feature_out, self).__init__()

        out = ['conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out']
        base_model = tf.keras.applications.resnet.ResNet152(input_shape=(240, 240, 3),
                                                            include_top=False,
                                                            weights='imagenet',
                                                            layer=tf.keras.layers)
        self.backbone_detector = tf.keras.Model(inputs=base_model.input,
                                                outputs=[base_model.get_layer(x).output for x in out])
        self.feature_merger = feature_merging.featrue_merging(fec)

        self.score = tf.keras.layers.Conv2D(1, kernel_size=(1, 1),
                                            activation=tf.nn.sigmoid,
                                            kernel_regularizer=None)
        self.geo_map = tf.keras.layers.Conv2D(4, kernel_size=(1, 1),
                                              activation=tf.nn.sigmoid,
                                              kernel_regularizer=None)
        self.rotation = tf.keras.layers.Conv2D(1, kernel_size=(1, 1),
                                               activation=tf.nn.sigmoid,
                                               kernel_regularizer=None)

    def call(self, inputs):
        C2, C3, C4, C5 = self.backbone_detector(inputs)
        feature_extractor = self.feature_merger(C2, C3, C4, C5)
        H_out = feature_extractor[0]

        # pass H_out through the score layer
        F_score = self.score(H_out)

        # obtain F_geometry in RBOX manner
        geo_map = self.geo_map(H_out)
        rotation = self.rotation(H_out)

        F_geometry = tf.concat([geo_map, rotation], axis=-1)

        pred = {'F_score': F_score,
                'F_geometry': F_geometry}

        return pred
