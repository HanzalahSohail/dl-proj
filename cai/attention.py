# In file: k/cai/attention.py

import tensorflow as tf
from tensorflow.keras import layers, models

@tf.keras.utils.register_keras_serializable(package='Cai')
class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = layers.Dense(channel // self.ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        self.shared_layer_two = layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # AvgPool
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)
        # MaxPool
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, max_pool.shape[1]))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)
        # Combine
        cbam_feature = layers.Add()([avg_pool, max_pool])
        cbam_feature = layers.Activation('sigmoid')(cbam_feature)
        return inputs * cbam_feature

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

@tf.keras.utils.register_keras_serializable(package='Cai')
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv2d = layers.Conv2D(filters=1,
                                    kernel_size=self.kernel_size,
                                    strides=1,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer='he_normal',
                                    use_bias=False)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        spatial_feature = self.conv2d(concat)
        return inputs * spatial_feature

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

@tf.keras.utils.register_keras_serializable(package='Cai')
class CBAMBlock(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio=self.ratio)
        self.spatial_attention = SpatialAttention(kernel_size=self.kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_size': self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable(package='Cai')
class CrossAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        # You can also use tf.keras.layers.MultiHeadAttention directly
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.d_model * 4, activation='relu'),
            layers.Dense(self.d_model)
        ])

    def call(self, feat_q, feat_kv):
        # reshape to sequences: B, H*W, C
        B, H, W, C = tf.shape(feat_q)[0], tf.shape(feat_q)[1], tf.shape(feat_q)[2], tf.shape(feat_q)[3]
        seq_len = H * W
        q = tf.reshape(feat_q, (B, seq_len, C))
        kv = tf.reshape(feat_kv, (B, seq_len, C))

        # cross-attention
        attn_out = self.mha(query=q, value=kv, key=kv)  
        attn_out = self.norm1(q + attn_out)

        # feed‑forward
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)

        # back to feature‑map shape
        return tf.reshape(out, (B, H, W, C))