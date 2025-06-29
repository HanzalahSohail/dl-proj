# In file: k/cai/attention.py

import tensorflow as tf
from tensorflow.keras import layers, initializers

# --- HELPER FUNCTION MOVED TO THE TOP-LEVEL ---
# By being a global function in the module, Keras can find it during model loading.
def spatial_output_shape(input_shape):
    """
    Calculates the output shape for spatial attention, which is (batch, height, width, 1).
    """
    return input_shape[:-1] + (1,)
# --------------------------------------------

def channel_attention(input_feature, ratio=8):
    """
    Applies Channel Attention to the input feature map.
    """
    channel = input_feature.shape[-1]
    
    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    
    return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size=7):
    """
    Applies Spatial Attention to the input feature map.
    """
    cbam_feature = input_feature

    # The Lambda layers now correctly reference the top-level helper function
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True),
                             output_shape=spatial_output_shape)(cbam_feature)
                             
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True),
                             output_shape=spatial_output_shape)(cbam_feature)

    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    
    cbam_feature = layers.Conv2D(filters=1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid',
                                 kernel_initializer='he_normal',
                                 use_bias=False)(concat)
    
    return layers.multiply([input_feature, cbam_feature])

def cbam_block(cbam_feature, ratio=8):
    """
    The complete Convolutional Block Attention Module (CBAM).
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature
