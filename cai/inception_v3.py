"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567) (CVPR 2016)

COPYRIGHT

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this file came from https://github.com/keras-team/keras-applications/edit/master/keras_applications/inception_v3.py
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.

The original file from above link was modified. Modifications can be tracked via 
git commits at https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/inception_v3.py

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from . import attention 
import cai.layers
from cai.layers import conv2d_bn
import cai.util
from tensorflow import keras
from tensorflow.keras.models import Model
from . import color_layers

def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = keras.layers.Input(shape=input_shape)

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = keras.layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = keras.layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = keras.models.Model(inputs, x, name='inception_v3')

    return model

def kInceptionPointwise(last_tensor, filters=32, channel_axis=3, name=None, activation='relu', has_batch_norm=True, has_batch_scale=False, use_bias=False, kType=0):
  return cai.layers.kPointwiseConv2D(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kType=kType)

def create_inception_v3_mixed_layer(x,  id,  name='', channel_axis=3, bottleneck_compression=1,  compression=1, kType=0):
    if id == 0:
            # mixed 0: 35 x 35 x 256
            # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
            branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*64), name=name + '_11a', kType=kType)
            # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
            branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression*48), name=name + '_11b', kType=kType)
            branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
            # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
            branch3x3dbl = kInceptionPointwise(x, filters=int(compression*64), name=name + '_11c')
            branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
            branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33cc')
            branch_pool = keras.layers.AveragePooling2D((3, 3),strides=(1, 1),padding='same', name=name + '_avg')(x)
            # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*32), 1, 1, name=name + '_avg11')
            branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*32), name=name + '_avg11', kType=kType)
            x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)
    
    if id == 1:
        # mixed 1: 35 x 35 x 288
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*64), name=name + '_11a', kType=kType)
        # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression*48), name=name + '_11b', kType=kType)
        branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
        # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = kInceptionPointwise(x, filters=int(compression*64), name=name + '_11c', kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*64), name=name + '_avg11', kType=kType)
        x = keras.layers.concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 2:
        # mixed 2: 35 x 35 x 288
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*64), name=name + '_11a', kType=kType)
        # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression*48), name=name + '_11b', kType=kType)
        branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
        # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = kInceptionPointwise(x, filters=int(compression*64), name=name + '_11c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33bb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*64), name=name + '_avg11')
        x = keras.layers.concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 3:
        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, int(bottleneck_compression*384), 3, 3, strides=(2, 2), padding='valid', name=name + '_33a')
        # branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11b')
        branch3x3dbl = kInceptionPointwise(x, filters=int(bottleneck_compression*64), name=name + '_11b', kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33b')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, strides=(2, 2), padding='valid')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name=name + '_max')(x)
        x = keras.layers.concatenate( [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 4:
        # mixed 4: 17 x 17 x 768
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11a', kType=kType)
        # branch7x7 = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression*128), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression*128), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        # branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression*128), name=name + '_11c', kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*192), name=name + '_avg11', kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if ((id == 5) or (id == 6)):
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11a', kType=kType)
        # branch7x7 = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression*160), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression*160), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        # branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression*160), name=name + '_11c', kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D( (3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*192), name=name + '_avg11', kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 7:
        # mixed 7: 17 x 17 x 768
        #branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11a', kType=kType)
        #branch7x7 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        #branch7x7dbl = conv2d_bn(x, int(compression*192), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11c', kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*192), name=name + '_avg11', kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 8:
        # mixed 8: 8 x 8 x 1280
        # branch3x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch3x3 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11a', kType=kType)
        branch3x3 = conv2d_bn(branch3x3, int(compression*320), 3, 3, strides=(2, 2), padding='valid', name=name + '_33a')
        # branch7x7x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7x3 = kInceptionPointwise(x, filters=int(bottleneck_compression*192), name=name + '_11b', kType=kType)
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 1, 7, name=name + '_17b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 7, 1, name=name + '_71b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 3, 3, strides=(2, 2), padding='valid', name=name + '_33b')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),  name=name + '_max')(x)
        x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name=name)

    if (id == 9) or (id==10):
        # mixed 9: 8 x 8 x 2048
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*320), 1, 1, name=name + '_11')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression*320), name=name + '_11', kType=kType)
        # branch3x3 = conv2d_bn(x, int(bottleneck_compression*384), 1, 1, name=name + '_11a')
        branch3x3 = kInceptionPointwise(x, filters=int(bottleneck_compression*384), name=name + '_22', kType=kType)
        branch3x3_1 = conv2d_bn(branch3x3, int(compression*384), 1, 3, name=name + '_11a')
        branch3x3_2 = conv2d_bn(branch3x3, int(compression*384), 3, 1, name=name + '_31a')
        branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name=name + '_pa')
        # branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*448), 1, 1, name=name + '_11b')
        branch3x3dbl = kInceptionPointwise(x, filters=int(bottleneck_compression*448), name=name + '_11b', kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*384), 3, 3, name=name + '_33b')
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, int(compression*384), 1, 3, name=name + '_13b')
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, int(compression*384), 3, 1, name=name + '_31b')
        branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis, name=name + '_pb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression*192), name=name + '_avg11', kType=kType)
        x = keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)
    return x

def create_inception_path(last_tensor,  compression=0.5,  channel_axis=3,  name=None, activation=None, has_batch_norm=True, kType=0):
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    channel_count = int(prev_layer_channel_count * compression)
    # output_tensor = conv2d_bn(output_tensor, channel_count, 1, 1, name=name, activation=activation, has_batch_norm=has_batch_norm)        
    output_tensor = kInceptionPointwise(output_tensor, filters=channel_count, name=name, activation=activation, has_batch_norm=has_batch_norm, kType=kType)
    return output_tensor

def create_inception_v3_two_path_mixed_layer(x, id, name='', channel_axis=3, bottleneck_compression=0.5, compression=0.655, has_batch_norm=False, kType=0):
    if name=='':
        name='mixed'
    interleaved  = cai.layers.InterleaveChannels(2,  name=name+'_interleaved')(x)
    a = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis, name=name+'_ta', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    b = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis, name=name+'_tb', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    a = create_inception_v3_mixed_layer(a, id=id, name=name+'a', bottleneck_compression=bottleneck_compression, compression=compression, kType=kType)
    b = create_inception_v3_mixed_layer(b, id=id, name=name+'b', bottleneck_compression=bottleneck_compression, compression=compression, kType=kType)
    return keras.layers.Concatenate(axis=channel_axis, name=name)([a, b])

def two_path_inception_v3(
                include_top=True,
                weights=None, #'two_paths_plant_leafs'
                input_shape=(224,224,3),
                pooling=None,
                classes=1000,
                two_paths_partial_first_block=0,
                two_paths_first_block=False,
                two_paths_second_block=False,
                deep_two_paths=False,
                deep_two_paths_compression=0.655,
                deep_two_paths_bottleneck_compression=0.5,
                l_ratio=0.5,
                ab_ratio=0.5,
                max_mix_idx=10,
                max_mix_deep_two_paths_idx=-1,
                model_name='two_path_inception_v3',
                kType=0,
                **kwargs):
    """Instantiates the Inception v3 architecture with 2 paths options.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_shape: mandatory input shape. Common values are 
            (299, 299, 3) and (224, 224, 3).
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        two_paths_partial_first_block: valid values are 1, 2 and 3. 1 means
            only one two-paths convolution. 2 means 2 two-paths convolutions. 3 means
            a full first two-path block. Other values mean nothing.
        two_paths_first_block: when true, starts with 2 paths for 
            the first 3 convolutions.
        two_paths_second_block: when true, another 2 convolutions
            are done in two paths.
        deep_two_paths: when true, creates a complete two-path architecture.
        deep_two_paths_compression: how much each path should be compressed.
        l_ratio: proportion dedicated to light.
        ab_ratio: proportion dedicated to color.
        max_mix_idx: last "mixed layer" index. You can create smaller
            architectures with this parameter.
        max_mix_deep_two_paths_idx: last "mixed layer" index with two-paths.
        model_name: model name
        kType: k optimized convolutional type.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = keras.layers.Input(shape=input_shape, name='rgb_input')

    # --- Path 1: RGB Branch ---
    # Process the raw RGB input directly
    print("Creating RGB Path...")
    rgb_branch = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    rgb_branch = conv2d_bn(rgb_branch, 32, 3, 3, padding='valid')
    rgb_branch = conv2d_bn(rgb_branch, 64, 3, 3)
    rgb_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(rgb_branch)
    rgb_branch = conv2d_bn(rgb_branch, 80, 1, 1, padding='valid')
    rgb_branch = conv2d_bn(rgb_branch, 192, 3, 3, padding='valid')
    rgb_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(rgb_branch)

    # --- Convert to LAB and create other paths ---
    print("Creating LAB Paths...")
    # Convert the RGB input to LAB inside the model
    lab_tensor = color_layers.RgbToLab(name='rgb_to_lab')(img_input)

    # --- Path 2: L (Lightness) Branch ---
    l_branch = cai.layers.CopyChannels(0, 1, name='copy_L_channel')(lab_tensor)
    l_branch = conv2d_bn(l_branch, 32, 3, 3, strides=(2, 2), padding='valid')
    l_branch = conv2d_bn(l_branch, 32, 3, 3, padding='valid')
    l_branch = conv2d_bn(l_branch, 64, 3, 3)
    l_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(l_branch)
    l_branch = conv2d_bn(l_branch, 80, 1, 1, padding='valid')
    l_branch = conv2d_bn(l_branch, 192, 3, 3, padding='valid')
    l_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(l_branch)

    # --- Path 3: a*b* (Color) Branch ---
    ab_branch = cai.layers.CopyChannels(1, 2, name='copy_ab_channels')(lab_tensor)
    ab_branch = conv2d_bn(ab_branch, 32, 3, 3, strides=(2, 2), padding='valid')
    ab_branch = conv2d_bn(ab_branch, 32, 3, 3, padding='valid')
    ab_branch = conv2d_bn(ab_branch, 64, 3, 3)
    ab_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(ab_branch)
    ab_branch = conv2d_bn(ab_branch, 80, 1, 1, padding='valid')
    ab_branch = conv2d_bn(ab_branch, 192, 3, 3, padding='valid')
    ab_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(ab_branch)

    # --- Concatenate all three paths ---
    print("Concatenating RGB, L, and a*b* paths...")
    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = keras.layers.Concatenate(axis=channel_axis, name='concatenate_all_paths')([rgb_branch, l_branch, ab_branch])
    if max_mix_idx >= 0:
        for id_layer in range(max_mix_idx+1):
            if (max_mix_deep_two_paths_idx >= id_layer):
                x = create_inception_v3_two_path_mixed_layer(x,  id=id_layer,  name='mixed'+str(id_layer),
                    channel_axis=channel_axis, bottleneck_compression=deep_two_paths_bottleneck_compression, 
                    compression=deep_two_paths_compression, has_batch_norm=True, kType=kType)
            else:
                x = create_inception_v3_mixed_layer(x,  id=id_layer,  name='mixed'+str(id_layer), channel_axis=channel_axis, kType=kType)

            # --- ADDING THE FOLLOWING IF-BLOCK ---
            # Add the CBAM block after the 'mixed5' layer has been created.
            if id_layer == 5:
                print("Applying CBAM block after mixed5...")
                # x = attention.cbam_block(x)
                x = attention.CBAMBlock()(x)
            # ------------------------------------
    if include_top:
        # Classification block
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = keras.models.Model(inputs, x, name=model_name)
    return model

def compiled_full_two_path_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    max_mix_idx=10, 
    model_name='two_path_inception_v3',
    optimizer=None):
    """Returns a compiled full two-paths inception v3.
    # Arguments
        input_shape: mandatory input shape. Common values are 
            (299, 299, 3) and (224, 224, 3).
        classes: number of classes to classify images into.
        max_mix_idx: last "mixed layer" index. You can create smaller
            architectures with this parameter.
        model_name: model name
        optimizer: if present, is the optimizer used for compilation.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    return compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=True,
        two_paths_second_block=True,
        deep_two_paths=True,
        deep_two_paths_compression=0.655,
        max_mix_idx=max_mix_idx, 
        model_name='deep_two_path_inception_v3',
        optimizer=optimizer
    )
    
def compiled_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    max_mix_idx=10, 
    model_name='two_path_inception_v3',
    optimizer=None):
    """Returns a compiled two-paths inception v3.
    # Arguments
        input_shape: mandatory input shape. Common values are 
            (299, 299, 3) and (224, 224, 3).
        classes: number of classes to classify images into.
        max_mix_idx: last "mixed layer" index. You can create smaller
            architectures with this parameter.
        model_name: model name
        optimizer: if present, is the optimizer used for compilation.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    return compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        max_mix_idx=max_mix_idx, 
        model_name='two_path_inception_v3',
        optimizer=optimizer
    )

def compiled_two_path_inception_v3(
    input_shape=(224,224,3),
    classes=1000,
    two_paths_partial_first_block=0,
    two_paths_first_block=False,
    two_paths_second_block=False,
    deep_two_paths=False,
    deep_two_paths_compression=0.655,
    deep_two_paths_bottleneck_compression=0.5,
    l_ratio=0.5,
    ab_ratio=0.5,
    max_mix_idx=10,
    max_mix_deep_two_paths_idx=-1,
    model_name='two_path_inception_v3', 
    optimizer=None
    ):
    """Returns a compiled two-paths inception v3.
    # Arguments
        input_shape: mandatory input shape. Common values are 
            (299, 299, 3) and (224, 224, 3).
        classes: number of classes to classify images into.
        two_paths_partial_first_block: valid values are 1, 2 and 3. 1 means
            only one two-paths convolution. 2 means 2 two-paths convolutions. 3 means
            a full first two-path block. Other values mean nothing.
        two_paths_first_block: when true, starts with 2 paths for 
            the first 3 convolutions.
        two_paths_second_block: when true, another 2 convolutions
            are done in two paths.
        deep_two_paths: when true, creates a complete two-path architecture.
        deep_two_paths_compression: how much each path should be compressed.
        l_ratio: proportion dedicated to light.
        ab_ratio: proportion dedicated to color.
        max_mix_idx: last "mixed layer" index. You can create smaller
            architectures with this parameter.
        max_mix_deep_two_paths_idx: last "mixed layer" index with two-paths.
        model_name: model name 
        optimizer: if present, is the optimizer used for compilation.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    base_model = two_path_inception_v3(
        include_top=False, # Has to be false to be compiled below.
        weights=None,
        input_shape=input_shape,
        pooling=None, # Has to be none to be compiled below.
        classes=classes,
        two_paths_partial_first_block=two_paths_partial_first_block,
        two_paths_first_block=two_paths_first_block,
        two_paths_second_block=two_paths_second_block,
        deep_two_paths=deep_two_paths,
        deep_two_paths_compression=deep_two_paths_compression,
        deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
        l_ratio=l_ratio,
        ab_ratio=ab_ratio,
        max_mix_idx=max_mix_idx,
        max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
        model_name=model_name
    )
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(classes, name='preprediction')(x)
    predictions = keras.layers.Activation('softmax',name='prediction')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if optimizer is None:
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    else:
        opt = optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    return model
