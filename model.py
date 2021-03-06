from keras.models import Model,Input
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense,Dropout,Flatten,Activation
from keras.layers.merge import concatenate,add
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU

def repeated_block(x,filters,kernel_size=3,pooling_size=3,dropout=0.5,is_first_layer_of_block=False,activation=LeakyReLU()):
    """
    residual block using pre activation
    :param x:
    :param filters:
    :param kernel_size:
    :param pooling_size:
    :param dropout:
    :param is_first_layer_of_block:
    :return:
    """

    k1,k2 = filters
    # program control it
    out = BatchNormalization()(x)
    out = Activation(activation)(out)
    out = Conv1D(k1,kernel_size,strides=1,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation(activation)(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=1,padding='same')(out)

    if is_first_layer_of_block:
        # add conv here
        pooling = Conv1D(k2,kernel_size,strides=1,padding='same')(x)
    else:
        pooling = MaxPooling1D(pooling_size, strides=1, padding='same')(x)
        pass

    out = add([out, pooling])


    return out



def build_residual_model(signal_timestep,
                         signal_dimension,
                         catalog,
                         number_conf,
                         output_dim,
                         block_number=8,
                         dropout=0.2,
                         activation='relu'):
    '''
    build residual neural network for NASA data
    :param activation:
    :param signal_timestep:
    :param signal_dimension:
    :param catalog:
    :param number_conf:
    :return:
    '''
    signal_input = Input(shape=(signal_timestep,signal_dimension),name='signal_input')
    catalog_input = Input(shape=(catalog,),name='material_one_hot_input')
    number_input = Input(shape=(number_conf,),name='conf_number')

    block_part_num = 5
    base_filter = 8
    output = signal_input

    total_times = block_number // block_part_num
    for cur_layer_num in range(block_number):
        is_first_layer = False
        if cur_layer_num % block_part_num == 0:
            is_first_layer = True
        # determine kernel size
        filter_times = total_times -  cur_layer_num // block_part_num
        filter = (base_filter*(2**(filter_times)),base_filter*(2**(filter_times)))
        print(filter,block_number,filter_times)
        print(cur_layer_num)
        output = repeated_block(output, filter, dropout=dropout,is_first_layer_of_block=is_first_layer,activation=activation)

    output = Flatten()(output)
    output = concatenate([output, catalog_input, number_input])
    output = BatchNormalization()(output)
    output = Activation(activation)(output)
    output = Dense(output_dim)(output)


    # output = BatchNormalization()(output)
    # output = Activation(activation)(output)
    # output = Dense(output_dim)(output)

    model = Model(inputs=[signal_input,catalog_input,number_input],outputs=output)

    optimizer = Adam(lr=1e-4)

    model.compile(optimizer,'mse',['mae'])

    return model



