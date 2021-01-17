import tensorflow as tf
from keras import backend as K
from keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, \
    Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

# from lib.segmentation.group_norm import GroupNormalization

smooth = 1.


class weighted_model:

    def complex_loss(self, mask):

        def weighted_binary_cross_entropy(y_true, y_pred):

            loss_alpha = 0.7
            dice_loss = -K.mean(self.dice_coef(y_true, y_pred))

            size_of_A_intersect_B = K.sum(y_true * y_pred * mask)
            size_of_A = K.sum(y_true * mask)
            size_of_B = K.sum(y_pred * mask)
            sign_B = tf.where(tf.greater(y_pred, 0), K.ones_like(mask), K.zeros_like(mask))
            if tf.greater(size_of_A_intersect_B, 0) is not None:
                c = K.sum(y_true * y_pred) / K.sum(y_true * sign_B)
            else:
                c = 1

            cDC = -K.mean((2. * size_of_A_intersect_B) + smooth) / ((c * size_of_A) + size_of_B + smooth)
            tf.summary.scalar("cdc_loss", (1 - loss_alpha) * cDC)
            tf.summary.scalar("dc_loss", loss_alpha * dice_loss)
            return (1 - loss_alpha) * cDC + loss_alpha * dice_loss

        return weighted_binary_cross_entropy

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    def downLayer(self, inputLayer, filterSize, i, bn=False, axis=4):

        conv = Conv2D(filterSize, (3, 3), activation='relu', padding='same', name='conv' + str(i) + '_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv2D(filterSize * 2, (3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)

        return pool, conv

    def upLayer(self, inputLayer, concatLayer, filterSize, i, bn=False, do=False):
        up = Conv2DTranspose(filterSize, (2, 2), strides=(2, 2), activation='relu', padding='same',
                             name='up' + str(i))(inputLayer)
        # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv2D(int(filterSize / 2), (3, 3), activation='relu', padding='same', name='conv' + str(i) + '_1')(
            up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv, training=True)
        conv = Conv2D(int(filterSize / 2), (3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(
            conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv

    def build_model(self, img_shape=(192, 256, 3), learning_rate=5e-5, gpu_id=None, nb_gpus=None, trained_model=None):

        input_img = Input(img_shape, name='img_inp')

        kernel_init = 'he_normal'
        sfs = 16  # start filter size
        bn = True
        do = True
        conv1, conv1_b_m = self.downLayer(input_img, sfs, 1, bn)
        conv2, conv2_b_m = self.downLayer(conv1, sfs * 2, 2, bn)

        conv3 = Conv2D(sfs * 4, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(sfs * 8, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(sfs * 16, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_1')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4 = Dropout(0.5, seed=4, name='Dropout_' + str(4))(conv4, training=True)
        conv4 = Conv2D(sfs * 16, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_2')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        # conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv2DTranspose(sfs * 16, (2, 2), strides=(2, 2), activation='relu', padding='same',
                              name='up' + str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv2D(int(sfs * 8), (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(5) + '_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed=5, name='Dropout_' + str(5))(conv5, training=True)
        conv5 = Conv2D(int(sfs * 8), (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(5) + '_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer(conv5, conv2_b_m, sfs * 8, 6, bn, do)
        conv7 = self.upLayer(conv6, conv1_b_m, sfs * 4, 7, bn, do)

        conv_out = Conv2D(2, (1, 1), activation='softmax', name='conv_final')(conv7)
        bg = Lambda(lambda x: x[:, :, :, 0], name='bg')(conv_out)
        skin = Lambda(lambda x: x[:, :, :, 1], name='skin')(conv_out)

        # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        optimizer = Adam(lr=learning_rate)  # TODO: settings of optimizer
        p_model = Model(input_img, [bg, skin])
        p_model.compile(optimizer=optimizer, loss={'bg': self.dice_loss,
                                                   'skin': self.dice_loss},
                        metrics={'bg': self.dice_coef, 'skin': self.dice_coef})

        return p_model
