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
        # The input RGB values are expected to be in the range [0, 1].
        # We need to convert them to integers in the range [0, 255] for the conversion function.
        rgb_int = tf.image.convert_image_dtype(inputs, dtype=tf.uint8)
        
        # Convert from RGB to LAB.
        lab_image = tfio.experimental.color.rgb_to_lab(rgb_int)
        
        # The output of the conversion is float, so we cast it back to the input's dtype.
        return tf.cast(lab_image, inputs.dtype)

    def get_config(self):
        # Required for model saving and loading.
        return super(RgbToLab, self).get_config()