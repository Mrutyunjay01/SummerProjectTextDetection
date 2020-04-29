# for the time - being, proceed to test with resnet-152
import tensorflow as tf
import tensorflow.keras.layers as L


def _pad_and_add(x1, x2):
    """
    for parameters to concatenate with matched shape
    :param x1: an numpy array
    :param x2: another numpy array
    :return: Added after being matched shape
    """
    x1_shape = x1.shape
    x2_shape = x2.shape
    offsets = [0, (x2_shape[1] - x1_shape[1]) // 2, (x2_shape[2] - x1_shape[2]) // 2, 0]
    x1_pad = tf.pad(x1, offsets, size=x2.shape)
    return tf.add(x1_pad, x2)


class featrue_merging(tf.keras.layers):
    def __init__(self, inputs):
        super(featrue_merging, self).__init()

        self.unpool = tf.image.resize_with_crop_or_pad(inputs, tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2)

    def call(self, C2, C3, C4, C5):
        H1 = tf.keras.layers.Conv2D(C5.shape[3] / 4,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    strides=1,
                                    kernel_initializer=tf.keras.initializers.glorot_uniform())(C5)
        H1 = self.unpool(H1)
        H2 = _pad_and_add(H1, C4)
        H2 = tf.keras.layers.Conv2D(H2.shape[3]/4,
                                    kernel_size=(1, 1),
                                    kernel_initializer=tf.keras.initializers.glorot_uniform())(H2)
        H2 = tf.keras.layers.Conv2D(H2.shape[3],
                                    kernel_size=(3, 3),
                                    kernel_initializer=tf.keras.initializers.glorot_uniform())(H2)
        H2 = self.unpool(H2)
        H3 = _pad_and_add(H2, C3)
        H3 = L.Conv2D(H3.shape[3]/4,
                      kernel_size=(1, 1),
                      kernel_initializer=tf.keras.initializers.glorot_uniform())(H3)
        H3 = L.Conv2D(H3.shape[3],
                      kernel_size=(3, 3),
                      kernel_initializer=tf.keras.initializers.glorot_uniform())(H3)
        H3 = self.unpool(H3)
        H4 = _pad_and_add(H3, C2)
        H4 = L.Conv2D(H4.shape[3]/4,
                      kernel_size=(1, 1),
                      kernel_initializer=tf.keras.initializers.glorot_uniform())(H4)
        H4 = L.Conv2D(H4.shape[3],
                      kernel_size=(3, 3),
                      kernel_initializer=tf.keras.initializers.glorot_uniform())(H4)
        H4_out = L.Conv2D(H4.shape[3]/2,
                          kernel_size=(3, 3),
                          kernel_initializer=tf.keras.initializers.glorot_uniform())(H4)

        return H4_out

