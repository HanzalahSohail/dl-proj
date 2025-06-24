
import cai.layers
from cai.layers import conv2d_bn
import cai.util
from tensorflow import keras
from tensorflow.keras.models import Model
from cbam import cbam_block

def create_inception_v3_mixed_layer(x,  id,  name='', channel_axis=3, bottleneck_compression=1,  compression=1, kType=0):
    if id == 0:
            # mixed 0: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
            branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
            branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
            branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
            branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
            branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33cc')
            branch_pool = keras.layers.AveragePooling2D((3, 3),strides=(1, 1),padding='same', name=name + '_avg')(x)
            branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*32), 1, 1, name=name + '_avg11')
            x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)
    
    if id == 1:
        # mixed 1: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
        branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 2:
        # mixed 2: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = conv2d_bn(branch5x5, int(compression*64), 5, 5, name=name + '_55b')
        branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33bb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 3:
        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, int(bottleneck_compression*384), 3, 3, strides=(2, 2), padding='valid', name=name + '_33a')
        branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11b')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, name=name + '_33b')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*96), 3, 3, strides=(2, 2), padding='valid')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name=name + '_max')(x)
        x = keras.layers.concatenate( [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 4:
        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch7x7 = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*128), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*128), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if ((id == 5) or (id == 6)):
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch7x7 = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*160), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*160), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D( (3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 7:
        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch7x7 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression*192), 7, 1, name=name + '_71b')
        branch7x7dbl = conv2d_bn(x, int(compression*192), 1, 1, name=name + '_11c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression*192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 8:
        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch3x3 = conv2d_bn(branch3x3, int(compression*320), 3, 3, strides=(2, 2), padding='valid', name=name + '_33a')
        branch7x7x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 1, 7, name=name + '_17b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 7, 1, name=name + '_71b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression*192), 3, 3, strides=(2, 2), padding='valid', name=name + '_33b')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2),  name=name + '_max')(x)
        x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name=name)

    if (id == 9) or (id==10):
        # mixed 9: 8 x 8 x 2048
        branch1x1 = conv2d_bn(x, int(bottleneck_compression*320), 1, 1, name=name + '_11')
        branch3x3 = conv2d_bn(x, int(bottleneck_compression*384), 1, 1, name=name + '_11a')
        branch3x3_1 = conv2d_bn(branch3x3, int(compression*384), 1, 3, name=name + '_11a')
        branch3x3_2 = conv2d_bn(branch3x3, int(compression*384), 3, 1, name=name + '_31a')
        branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name=name + '_pa')
        branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*448), 1, 1, name=name + '_11b')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression*384), 3, 3, name=name + '_33b')
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, int(compression*384), 1, 3, name=name + '_13b')
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, int(compression*384), 3, 1, name=name + '_31b')
        branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis, name=name + '_pb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        x = keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)
    return x
def create_inception_v3_two_path_mixed_layer(x, id, name='', channel_axis=3, bottleneck_compression=0.5, compression=0.655, has_batch_norm=False, kType=0):
    if name=='':
        name='mixed'
    interleaved  = cai.layers.InterleaveChannels(2,  name=name+'_interleaved')(x)
    a = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis, name=name+'_ta', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    b = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis, name=name+'_tb', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    a = create_inception_v3_mixed_layer(a, id=id, name=name+'a', bottleneck_compression=bottleneck_compression, compression=compression, kType=kType)
    b = create_inception_v3_mixed_layer(b, id=id, name=name+'b', bottleneck_compression=bottleneck_compression, compression=compression, kType=kType)
    return keras.layers.Concatenate(axis=channel_axis, name=name)([a, b])
def two_path_inception_v3_with_cbam(
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
                use_cbam=False, # New parameter
                **kwargs):
    img_input = keras.layers.Input(shape=input_shape)
    if (deep_two_paths):  max_mix_deep_two_paths_idx = max_mix_idx

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    if two_paths_partial_first_block==3:
        two_paths_partial_first_block=0
        two_paths_first_block=True
        two_paths_second_block=False

    if two_paths_partial_first_block>3:
        two_paths_partial_first_block=0
        two_paths_first_block=True
        two_paths_second_block=True

    if (two_paths_second_block):
        two_paths_first_block=True
    
    include_first_block=True
    if (two_paths_partial_first_block==1) or (two_paths_partial_first_block==2):
        two_paths_second_block=False
        two_paths_first_block=False
        include_first_block=False

        # Only 1 convolution with two-paths?
        if (two_paths_partial_first_block==1):
            if (l_ratio>0):
                l_branch = cai.layers.CopyChannels(0,1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, strides=(2, 2), padding='valid')

            if (ab_ratio>0):
                ab_branch = cai.layers.CopyChannels(1,2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, strides=(2, 2), padding='valid')

            if (l_ratio>0):
                if (ab_ratio>0):
                    single_branch  = keras.layers.Concatenate(axis=channel_axis, name='concat_partial_first_block1')([l_branch, ab_branch])
                else:
                    single_branch = l_branch
            else:
                single_branch = ab_branch

            single_branch = conv2d_bn(single_branch, 32, 3, 3, padding='valid')
            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)

        # Only 2 convolution with two-paths?
        if (two_paths_partial_first_block==2):
            if (l_ratio>0):
                l_branch = cai.layers.CopyChannels(0,1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, strides=(2, 2), padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, padding='valid')

            if (ab_ratio>0):
                ab_branch = cai.layers.CopyChannels(1,2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, strides=(2, 2), padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, padding='valid')

            if (l_ratio>0):
                if (ab_ratio>0):
                    single_branch = keras.layers.Concatenate(axis=channel_axis, name='concat_partial_first_block2')([l_branch, ab_branch])
                else:
                    single_branch = l_branch
            else:
                single_branch = ab_branch

            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)

    if include_first_block:
        if two_paths_first_block:
            if (l_ratio>0):
                l_branch = cai.layers.CopyChannels(0,1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, strides=(2, 2), padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(32*l_ratio)), 3, 3, padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(64*l_ratio)), 3, 3)
                l_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(l_branch)

            if (ab_ratio>0):
                ab_branch = cai.layers.CopyChannels(1,2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, strides=(2, 2), padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(32*ab_ratio)), 3, 3, padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(64*ab_ratio)), 3, 3)
                ab_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(ab_branch)
            
            if (l_ratio>0):
                if (ab_ratio>0):
                    x = keras.layers.Concatenate(axis=channel_axis, name='concat_first_block')([l_branch, ab_branch])
                else:
                    x = l_branch
            else:
                x = ab_branch
        else:
            single_branch = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
            single_branch = conv2d_bn(single_branch, 32, 3, 3, padding='valid')
            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            single_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)
            x = single_branch

    if (two_paths_second_block):
      l_branch    = conv2d_bn(x, int(round(80*deep_two_paths_bottleneck_compression)), 1, 1, padding='valid', name='second_block_ta', activation=None, has_batch_norm=True)
      ab_branch = conv2d_bn(x, int(round(80*deep_two_paths_bottleneck_compression)), 1, 1, padding='valid', name='second_block_tb', activation=None, has_batch_norm=True)
      
      l_branch    = conv2d_bn(l_branch,    int(round(80 *deep_two_paths_compression)), 1, 1, padding='valid')
      l_branch    = conv2d_bn(l_branch,    int(round(192*deep_two_paths_compression)), 3, 3, padding='valid')
      ab_branch = conv2d_bn(ab_branch, int(round(80 *deep_two_paths_compression)), 1, 1, padding='valid')
      ab_branch = conv2d_bn(ab_branch, int(round(192*deep_two_paths_compression)), 3, 3, padding='valid')
      
      x = keras.layers.Concatenate(axis=channel_axis, name='concat_second_block')([l_branch, ab_branch])
      x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    else:
      x = conv2d_bn(x, 80, 1, 1, padding='valid')
      x = conv2d_bn(x, 192, 3, 3, padding='valid')
      x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    if max_mix_idx >= 0:
        for id_layer in range(max_mix_idx+1):
            if (max_mix_deep_two_paths_idx >= id_layer):
                x = create_inception_v3_two_path_mixed_layer(x,  id=id_layer,  name='mixed'+str(id_layer),
                    channel_axis=channel_axis, bottleneck_compression=deep_two_paths_bottleneck_compression, 
                    compression=deep_two_paths_compression, has_batch_norm=True, kType=kType)
            else:
                x = create_inception_v3_mixed_layer(x,  id=id_layer,  name='mixed'+str(id_layer), channel_axis=channel_axis, kType=kType)
            if use_cbam:
                x = cbam_block(x) # Add CBAM block here
    
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
def compiled_two_path_inception_v3_with_cbam(
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
    model_name='two_path_inception_v3_with_cbam', 
    optimizer=None,
    use_cbam=False
    ):
    base_model = two_path_inception_v3_with_cbam(
        include_top=False, 
        weights=None,
        input_shape=input_shape,
        pooling=None, 
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
        model_name=model_name,
        use_cbam=use_cbam
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