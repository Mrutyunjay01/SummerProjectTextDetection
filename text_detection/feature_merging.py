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
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    x1_pad = tf.slice(x1, offsets, size=x2.shape)
    return tf.add(x1_pad, x2)


def unpool(X):
    return tf.image.resize_with_crop_or_pad(X, X.shape[1] * 2, X.shape[2] * 2)


class featrue_merging(tf.keras.layers.Layer):
    def __init__(self, fec):  # fec stands for feaure extractor channel
        super(featrue_merging, self).__init__()

        self.lateralConvC5 = L.Conv2D(fec * 8,
                                      kernel_size=(1, 1),
                                      kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H1_conv1 = L.Conv2D(fec * 8,
                                 kernel_size=(1, 1),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H1_conv2 = L.Conv2D(fec * 8,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConvC4 = L.Conv2D(fec * 4,
                                      kernel_size=(1, 1),
                                      kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H2_conv1 = L.Conv2D(fec * 4,
                                 kernel_size=(1, 1),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H2_conv2 = L.Conv2D(fec * 4,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConvC3 = L.Conv2D(fec * 2,
                                      kernel_size=(1, 1),
                                      kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H3_conv1 = L.Conv2D(fec * 2,
                                 kernel_size=(1, 1),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H3_conv2 = L.Conv2D(fec * 2,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConvC2 = L.Conv2D(fec,
                                      kernel_size=(1, 1),
                                      kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H4_conv1 = L.Conv2D(fec,
                                 kernel_size=(1, 1),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.H4_conv2 = L.Conv2D(fec,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.out = L.Conv2D(fec/2,
                            kernel_size=(3, 3),
                            kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, C2, C3, C4, C5):
        """
        :param C2: Output tensor from stage2 conv block of backbone detector
        :param C3: Output tensor from stage3 conv block of backbone detector
        :param C4: Output tensor from stage3 conv block of backbone detector
        :param C5: Output tensor from stage3 conv block of backbone detector
        :return: output a tensor of shape ( 3, 3, fec) for output layer of detection FCN
        """
        # similar to the paper ref: arXiv:1704.03155v2
        # extract the output of the last stage of the detector net
        H0 = self.lateralConv(C5)  #lateralConv can be defined and tested later
        # unpool to match with the shape
        H0 = unpool(H0)
        H1 = self.lateralConvC4(C4)
        H1 = _pad_and_add(H0, H1)
        H1 = self.H2_conv1(H1)
        H1 = self.H2_conv2(H1)

        # shape (None, None, None, fec * 4)
        H1 = unpool(H1)
        H2 = self.lateralConvC3(C3)
        H2 = _pad_and_add(H1, H2)
        H2 = self.H3_conv1(H2)
        H2 = self.H3_conv2(H2)

        # shape (None, None, None, fec * 2)
        H2 = unpool(H2)
        H3 = self.lateralConvC2(C2)
        H3 = _pad_and_add(H2, H3)
        H3 = self.H4_conv1(H3)
        H3 = self.H4_conv2(H3)

        # shape (None, None, None, fec)
        H_out = self.out(H3)

        return H_out
