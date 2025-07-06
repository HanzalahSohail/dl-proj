# In file: k/cai/color_layers.py

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package='Cai')
class RgbToLab(layers.Layer):
    """
    A Keras Layer to convert a batch of RGB images to the CIELAB color space.
    Input shape: (batch, height, width, 3) in RGB format.
    Output shape: (batch, height, width, 3) in LAB format.
    """
    def __init__(self, **kwargs):
        super(RgbToLab, self).__init__(**kwargs)

    def call(self, inputs):
        # The input tensor is already float32, which is what rgb_to_lab expects.
        # The incorrect conversion to uint8 has been removed.
        lab_image = tfio.experimental.color.rgb_to_lab(inputs)
        return lab_image

    def get_config(self):
        # Required for model saving and loading.
        return super(RgbToLab, self).get_config()