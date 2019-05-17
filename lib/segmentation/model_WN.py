import tensorflow as tf
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers import Conv2D, LeakyReLU, Dense
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.layers.core import Activation
from keras.models import Model


class MeanOnlyBatchNormalization(Layer):
    def __init__(self,
                 momentum=0.999,
                 moving_mean_initializer='zeros',
                 axis=-1,
                 **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.moving_mean_initializer = moving_mean_initializer
        self.axis = axis

    def build(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        super().build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        # inference
        def normalize_inference():
            return inputs - self.moving_mean

        if training in {0, False}:
            return normalize_inference()

        mean = K.mean(inputs, axis=reduction_axes)
        normed_training = inputs - mean

        self.add_update(K.moving_average_update(self.moving_mean, mean,
                                                self.momentum), inputs)

        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Bias(Layer):
    def __init__(self,
                 filters,
                 data_format=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        self.filters = filters
        self.data_format = data_format
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = K.bias_add(
            inputs,
            self.bias,
            data_format=self.data_format)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def WN_Conv2D(net, filters=None, kernel_size=None, padding='same', kernel_initializer='he_norml'):
    """Convolution layer with Weight Normalization + Mean-Only BatchNormalization"""
    net = Conv2D(filters, kernel_size, activation=None, padding=padding, kernel_initializer=kernel_initializer,
                 use_bias=False)(net)
    net = MeanOnlyBatchNormalization()(net)
    net = Bias(filters)(net)
    net = LeakyReLU(alpha=0.1)(net)

    return net


def WN_Dense(net, units=None, kernel_initializer='he_norml'):
    """Dense layer with Weight Normalization + Mean-Only BatchNormalization"""
    net = Dense(units=units, activation=None, kernel_initializer=kernel_initializer, use_bias=False)(net)
    net = MeanOnlyBatchNormalization()(net)
    net = Bias(units)(net)
    net = Activation(tf.nn.softmax)(net)

    return net

def downLayer(inputLayer, filterSize, i, bn=False):

    conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(inputLayer)
    if bn:
        conv = BatchNormalization()(conv)
    conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
    if bn:
        conv = BatchNormalization()(conv)
    pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)

    return pool, conv

def upLayer(inputLayer, concatLayer, filterSize, i, bn=False, do= False):

    up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',  name='up'+str(i))(inputLayer)
       # print( concatLayer.shape)
    up = concatenate([up, concatLayer])
    conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(up)
    if bn:
        conv = BatchNormalization()(conv)
    if do:
        conv = Dropout(0.5, seed = 3, name='Dropout_' + str(i))(conv)
    conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
    if bn:
        conv = BatchNormalization()(conv)

        return conv


def build_model(img_shape=(32, 168, 168), num_class=5, batch_num=2, learning_rate=5e-5):
    #input_img = Input(shape=(32, 32, 3))
    input_img = Input((*img_shape, 1), name='img_inp')
    supervised_label = Input(shape=(*img_shape, num_class), name='sup_label_inp')
    supervised_flag = Input(shape=(*img_shape, 1), name='sup_flag_inp')
    unsupervised_weight = Input(shape=(*img_shape, num_class), name='unsup_wt_inp')
    # input_idx = Input(shape=(batch_num))

    kernel_init = 'he_normal'
    sfs = 16  # start filter size
    bn = True
    do = True
    conv1, conv1_b_m = downLayer(input_img, sfs, 1, bn)
    conv2, conv2_b_m = downLayer(conv1, sfs * 2, 2, bn)

    conv3 = Conv3D(sfs * 4, (3, 3, 3), activation='relu', padding='same',  kernel_initializer = kernel_init, name='conv' + str(3) + '_1')(conv2)
    if bn:
        conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same',  kernel_initializer = kernel_init, name='conv' + str(3) + '_2')(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    # conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)

    conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same', kernel_initializer = kernel_init,  name='conv4_1')(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)
    if do:
        conv4 = Dropout(0.5, seed=4, name='Dropout_' + str(4))(conv4)
    conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same',  kernel_initializer = kernel_init, name='conv4_2')(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)

    # conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
    up1 = Conv3DTranspose(sfs * 16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same',
                          name='up' + str(5))(conv4)
    up1 = concatenate([up1, conv3])
    conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same',  kernel_initializer = kernel_init, name='conv' + str(5) + '_1')(up1)
    if bn:
        conv5 = BatchNormalization()(conv5)
    if do:
        conv5 = Dropout(0.5, seed=5, name='Dropout_' + str(5))(conv5)
    conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same',  kernel_initializer = kernel_init, name='conv' + str(5) + '_2')(conv5)
    if bn:
        conv5 = BatchNormalization()(conv5)

    conv6 = upLayer(conv5, conv2_b_m, sfs * 8, 6, bn, do)
    conv7 = upLayer(conv6, conv1_b_m, sfs * 4, 7, bn, do)

    conv_out = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax', kernel_initializer=kernel_init)(conv7)
    '''
    net = GaussianNoise(stddev=0.15)(input_img)

    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (3, 3), padding='same', kernel_initializer=kernel_init)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Dropout(rate=0.5)(net)

    net = WN_Conv2D(net, 512, (3, 3), padding='valid', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 256, (1, 1), padding='valid', kernel_initializer=kernel_init)
    net = WN_Conv2D(net, 128, (1, 1), padding='valid', kernel_initializer=kernel_init)
    net = GlobalAveragePooling2D()(net)
    net = WN_Dense(net, units=num_class, kernel_initializer=kernel_init)
    '''

    pz_out = Lambda(lambda x: K.reshape(x[:, :, :, :, 0], tf.convert_to_tensor([-1, *img_shape, 1])), name='pz_out')(
        conv_out)
    cz_out = Lambda(lambda x: K.reshape(x[:, :, :, :, 1], tf.convert_to_tensor([-1, *img_shape, 1])), name='cz_out')(
        conv_out)
    us_out = Lambda(lambda x: K.reshape(x[:, :, :, :, 2], tf.convert_to_tensor([-1, *img_shape, 1])), name='us_out')(
        conv_out)
    afs_out = Lambda(lambda x: K.reshape(x[:, :, :, :, 3], tf.convert_to_tensor([-1, *img_shape, 1])), name='afs_out')(
        conv_out)
    bg_out = Lambda(lambda x: K.reshape(x[:, :, :, :, 4], tf.convert_to_tensor([-1, *img_shape, 1])), name='bg_out')(
        conv_out)

    pz_gt = Lambda(lambda x: K.reshape(x[:, :, :, :, 0], tf.convert_to_tensor([-1, *img_shape, 1])), name='pz_gt')(
        supervised_label)
    cz_gt = Lambda(lambda x: K.reshape(x[:, :, :, :, 1], tf.convert_to_tensor([-1, *img_shape, 1])), name='cz_gt')(
        supervised_label)
    us_gt = Lambda(lambda x: K.reshape(x[:, :, :, :, 2], tf.convert_to_tensor([-1, *img_shape, 1])), name='us_gt')(
        supervised_label)
    afs_gt = Lambda(lambda x: K.reshape(x[:, :, :, :, 3], tf.convert_to_tensor([-1, *img_shape, 1])), name='afs_gt')(
        supervised_label)
    bg_gt = Lambda(lambda x: K.reshape(x[:, :, :, :, 4], tf.convert_to_tensor([-1, *img_shape, 1])), name='bg_gt')(
        supervised_label)

    pz_wt = Lambda(lambda x: K.reshape(x[:, :, :, :, 0], tf.convert_to_tensor([-1, *img_shape, 1])), name='pz_wt')(
        unsupervised_weight)
    cz_wt = Lambda(lambda x: K.reshape(x[:, :, :, :, 1], tf.convert_to_tensor([-1, *img_shape, 1])), name='cz_wt')(
        unsupervised_weight)
    us_wt = Lambda(lambda x: K.reshape(x[:, :, :, :, 2], tf.convert_to_tensor([-1, *img_shape, 1])), name='us_wt')(
        unsupervised_weight)
    afs_wt = Lambda(lambda x: K.reshape(x[:, :, :, :, 3], tf.convert_to_tensor([-1, *img_shape, 1])), name='afs_wt')(
        unsupervised_weight)
    bg_wt = Lambda(lambda x: K.reshape(x[:, :, :, :, 4], tf.convert_to_tensor([-1, *img_shape, 1])), name='bg_wt')(
        unsupervised_weight)

    flag = Lambda(lambda x: x[:, :, :, :, 0], name='sup_flag')(supervised_flag)
    # concatenate label
    pz = concatenate([pz_out, pz_gt, supervised_flag, pz_wt], name='pz')
    cz = concatenate([cz_out, cz_gt, supervised_flag, cz_wt], name='cz')
    us = concatenate([us_out, us_gt, supervised_flag, us_wt], name='us')
    afs = concatenate([afs_out, afs_gt, supervised_flag, afs_wt], name='afs')
    bg = concatenate([bg_out, bg_gt, supervised_flag, bg_wt], name='bg')

    model = Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])




    return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])
    #return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg, input_idx])
