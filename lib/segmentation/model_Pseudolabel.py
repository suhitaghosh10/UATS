import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model

from lib.segmentation.weight_norm import AdamWithWeightnorm

smooth = 1.


class weighted_model:
    class LossCallback(Callback):
        def __init__(self, loss_alpha, patience=10, ctr=0):
            self.loss_alpha = loss_alpha
            self.patience = patience
            self.ctr = ctr

        def on_epoch_end(self, epoch, patience=10, logs={}):
            patience = self.patience
            min_val = 0.60
            cur_val = self.loss_alpha

            print(cur_val)
            ctr = self.ctr
            # c1 = np.floormod(ctr, patience)) == 0
            # c2 = sess.run(tf.math.greater(cur_val, min_val))
            self.loss_alpha = cur_val - 0.025
            print(self.loss_alpha)

            ctr = self.ctr + 1
            self.ctr = ctr

        def get_alpha(self):
            return self.loss_alpha

    def complex_loss(self, mask):
        """custom loss function"""

        smooth = 1.0
        '''

        def loss(y_true, y_pred, axis=(-4, -3, -2) , smooth=1.):

          y_true_f = K.flatten(y_true)
          y_pred_f = K.flatten(y_pred)
          mask_f = K.flatten(mask)


          pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f , mask_f)
          pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, (mask_f * y_pred_f) )

          epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
          pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
          pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

          loss =  -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

          return K.mean(loss)
          '''

        # th = 0.55

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

    def c_dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        size_of_A_intersect_B = K.sum(y_true_f * y_pred_f)
        size_of_A = K.sum(y_true_f)
        size_of_B = K.sum(y_pred_f)
        sign_B = tf.where(tf.greater(y_pred_f, 0), K.ones_like(y_pred_f), K.zeros_like(y_pred_f))
        if tf.greater(size_of_A_intersect_B, 0) is not None:
            c = K.sum(y_true_f * y_pred_f) / K.sum(y_true_f * sign_B)
        else:
            c = 1

        return ((2. * size_of_A_intersect_B) + smooth) / ((c * size_of_A) + size_of_B + smooth)
        # downsampling, analysis path

    def downLayer(self, inputLayer, filterSize, i, bn=False, axis=4):

        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)

        return pool, conv

    def upLayer(self, inputLayer, concatLayer, filterSize, i, bn=False, do=False):
        up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',
                             name='up' + str(i))(inputLayer)
        # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv3D(int(filterSize / 2), (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_1')(
            up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv)
        conv = Conv3D(int(filterSize / 2), (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(
            conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv

    def build_model(self, img_shape=(32, 168, 168), learning_rate=5e-5, gpu_id=None, nb_gpus=None, trained_model=None):

        input_img = Input((*img_shape, 1), name='img_inp')
        yTrueInputs = Input((32, 168, 168, 5), name='mask')

        kernel_init = 'he_normal'
        sfs = 16  # start filter size
        bn = True
        do = True
        conv1, conv1_b_m = self.downLayer(input_img, sfs, 1, bn)
        conv2, conv2_b_m = self.downLayer(conv1, sfs * 2, 2, bn)

        conv3 = Conv3D(sfs * 4, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        # conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)

        conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_1')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4 = Dropout(0.5, seed=4, name='Dropout_' + str(4))(conv4)
        conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_2')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        # conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv3DTranspose(sfs * 16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same',
                              name='up' + str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(5) + '_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed=5, name='Dropout_' + str(5))(conv5)
        conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv' + str(5) + '_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer(conv5, conv2_b_m, sfs * 8, 6, bn, do)
        conv7 = self.upLayer(conv6, conv1_b_m, sfs * 4, 7, bn, do)

        conv_out = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        pz = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(conv_out)
        cz = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(conv_out)
        us = Lambda(lambda x: x[:, :, :, :, 2], name='us')(conv_out)
        afs = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(conv_out)
        bg = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(conv_out)

        pzw = Lambda(lambda x: x[:, :, :, :, 0], name='pzw')(yTrueInputs)
        czw = Lambda(lambda x: x[:, :, :, :, 1], name='czw')(yTrueInputs)
        usw = Lambda(lambda x: x[:, :, :, :, 2], name='usw')(yTrueInputs)
        afsw = Lambda(lambda x: x[:, :, :, :, 3], name='afsw')(yTrueInputs)
        bgw = Lambda(lambda x: x[:, :, :, :, 4], name='bgw')(yTrueInputs)

        # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        if (nb_gpus is None):
            p_model = Model([input_img, yTrueInputs], [pz, cz, us, afs, bg])
            if trained_model is not None:
                p_model.load_weights(trained_model)

            # model_copy = Model([input_img, unsupervised_label, supervised_flag, unsupervised_weight],[pz_out, cz_out, us_out, afs_out, bg_out])

            # intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

            p_model.compile(optimizer=optimizer,
                            loss={'pz': self.complex_loss(pzw),
                                  'cz': self.complex_loss(czw),
                                  'us': self.complex_loss(usw),
                                  'afs': self.complex_loss(afsw),
                                  'bg': self.complex_loss(bgw)},

                            metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                     'afs': self.dice_coef, 'bg': self.dice_coef}
                            )
        else:
            with tf.device(gpu_id):
                model = Model([input_img, yTrueInputs],
                              [pz, cz, us, afs, bg])
                if trained_model is not None:
                    model.load_weights(trained_model)

                p_model = multi_gpu_model(model, gpus=nb_gpus)

                p_model.compile(optimizer=optimizer,
                                loss={'pz': self.complex_loss(pzw),
                                      'cz': self.complex_loss(czw),
                                      'us': self.complex_loss(usw),
                                      'afs': self.complex_loss(afsw),
                                      'bg': self.complex_loss(bgw)},
                                metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                         'afs': self.dice_coef, 'bg': self.dice_coef}
                                )

        return p_model
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg, input_idx])
