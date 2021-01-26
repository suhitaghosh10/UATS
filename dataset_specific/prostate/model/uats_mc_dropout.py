import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.models import Model
#from keras.optimizers import Adam
from keras.utils import multi_gpu_model


from utility.weight_norm import AdamWithWeightnorm as Adam


class weighted_model:
    alpha = 0.6
    epoch_ctr = K.variable(0, name='epoch_ctr')

    class LossCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):
            weighted_model.epoch_ctr = epoch
            print(weighted_model.epoch_ctr)

    def dice_coef(self, y_true, y_pred, smooth=1.):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_tb(self, input, class_wt=1.):
        def dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            supervised_flag = input[1, :, :, :, :]
            unsupervised_gt = input[0, :, :, :, :]
            alpha = 0.6
            # unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            y_true_final = tf.where(tf.equal(supervised_flag, 2), unsupervised_gt, y_true)
            supervised_flag = tf.where(tf.equal(supervised_flag, 2), K.ones_like(supervised_flag), supervised_flag)
            y_true = y_true_final * supervised_flag
            y_pred = y_pred * supervised_flag

            intersection = K.sum(y_true * y_pred, axis=axis)
            y_true_sum = K.sum(y_true, axis=axis)
            y_pred_sum = K.sum(y_pred, axis=axis)

            sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
            if tf.greater(intersection, 0) is not None:
                c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
            else:
                c = 1

            return - K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))

        return dice_loss

    def c_reg_dice_loss(self, input):
        def dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            supervised_flag = input[1, :, :, :, :]
            y_true = y_true * supervised_flag
            y_pred = y_pred * supervised_flag

            intersection = K.sum(y_true * y_pred, axis=axis)
            y_true_sum = K.sum(y_true, axis=axis)
            y_pred_sum = K.sum(y_pred, axis=axis)

            sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
            if tf.greater(intersection, 0) is not None:
                c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
            else:
                c = 1

            return - K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))

        return dice_loss

    def c_dice_loss(self, y_true, y_pred, weight, smooth=1., axis=(1, 2, 3)):

        y_true = y_true * weight
        y_pred = y_pred * weight

        intersection = K.sum(y_true * y_pred, axis=axis)
        y_true_sum = K.sum(y_true, axis=axis)
        y_pred_sum = K.sum(y_pred, axis=axis)

        sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
        if tf.greater(intersection, 0) is not None:
            c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
        else:
            c = 1

        return - K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))

    def unsup_dice_tb(self, input, class_wt=1.):
        unsupervised_gt = input[0, :, :, :, :]

        # unsupervised_gt = unsupervised_gt / (1 -  (0.6** (self.epoch_ctr + 1)))

        def unsup_dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            y_true = unsupervised_gt
            y_pred = y_pred

            intersection = K.sum(y_true * y_pred, axis=axis)
            y_true_sum = K.sum(y_true, axis=axis)
            y_pred_sum = K.sum(y_pred, axis=axis)

            sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
            if tf.greater(intersection, 0) is not None:
                c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
            else:
                c = 1

            avg_dice_coef = K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))

            return 1 - avg_dice_coef

        return unsup_dice_loss

    def dice_loss(self, y_true, y_pred, weight, smooth=1., axis=(1, 2, 3)):

        y_true = y_true * weight
        y_pred = y_pred * weight

        intersection = K.sum(y_true * y_pred, axis=axis)
        y_true_sum = K.sum(y_true, axis=axis)
        y_pred_sum = K.sum(y_pred, axis=axis)

        avg_dice_coef = K.mean((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth))

        return - avg_dice_coef

    def unsup_dice_loss(self, y_true, y_pred, weight, smooth=1., axis=(1, 2, 3)):

        y_true = y_true
        y_pred = y_pred

        intersection = K.sum(y_true * y_pred, axis=axis)
        y_true_sum = K.sum(y_true, axis=axis)
        y_pred_sum = K.sum(y_pred, axis=axis)

        avg_dice_coef = K.mean((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth))

        return 1 - avg_dice_coef

    def unsup_c_dice_loss(self, y_true, y_pred, smooth=1., axis=(1, 2, 3)):

        intersection = K.sum(y_true * y_pred, axis=axis)
        y_true_sum = K.sum(y_true, axis=axis)
        y_pred_sum = K.sum(y_pred, axis=axis)

        sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
        if tf.greater(intersection, 0) is not None:
            c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
        else:
            c = 1

        avg_dice_coef = K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))

        return 1 - avg_dice_coef

    def semi_supervised_loss(self, input, unsup_loss_class_wt=1., alpha=0.6):

        def loss_func(y_true, y_pred):
            print(K.eval(self.epoch_ctr))

            unsupervised_gt = input[0, :, :, :, :]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            # unsupervised_gt = tf.where(K.greater(unsupervised_gt, K.constant(0.9)), K.ones_like(unsupervised_gt),K.zeros_like(unsupervised_gt))
            supervised_flag = input[1, :, :, :, :]
            # weight = input[:, :, :, :, 2]  # last elem are weights

            y_true_final = tf.where(tf.equal(supervised_flag, 2), unsupervised_gt, y_true)
            supervised_flag = tf.where(tf.equal(supervised_flag, 2), K.ones_like(supervised_flag), supervised_flag)
            supervised_loss = self.c_dice_loss(y_true_final, y_pred, supervised_flag)
            # unsupervised_loss = self.focal_loss(unsupervised_gt, y_pred, weight)
            unsupervised_loss = self.unsup_c_dice_loss(unsupervised_gt, y_pred)

            return supervised_loss + unsup_loss_class_wt * unsupervised_loss

        return loss_func

    def downLayer(self, inputLayer, filterSize, i, bn=False):
        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)

        return pool, conv

    def downLayer_MC(self, inputLayer, filterSize, i, bn=False):
        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same', name='conv_mc' + str(i) + '_1')(
            inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same', name='conv_mc' + str(i) + '_2')(
            conv)
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

    def upLayer_MC(self, inputLayer, concatLayer, filterSize, i, bn=False, do=False):
        up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',
                             name='up_mc' + str(i))(inputLayer)
        # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv3D(int(filterSize / 2), (3, 3, 3), activation='relu', padding='same',
                      name='conv_mc' + str(i) + '_1')(
            up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed=3, name='Dropout_mc' + str(i))(conv, training=True)
        conv = Conv3D(int(filterSize / 2), (3, 3, 3), activation='relu', padding='same',
                      name='conv_mc' + str(i) + '_2')(
            conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv

    def build_model(self, img_shape=(32, 168, 168), learning_rate=5e-5, gpu_id=None,
                    nb_gpus=None,
                    trained_model=None):
        input_img = Input((*img_shape, 1), name='img_inp')
        unsupervised_label = Input((*img_shape, 5), name='unsup_label_inp')
        supervised_flag = Input(shape=img_shape, name='flag_inp')

        kernel_init = 'he_normal'
        sfs = 16  # start filter size
        bn = True
        do = True

        # normal train- without MC

        #######################################################
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

        conv_out_sm = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        pz_sm_out = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(conv_out_sm)
        cz_sm_out = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(conv_out_sm)
        us_sm_out = Lambda(lambda x: x[:, :, :, :, 2], name='us')(conv_out_sm)
        afs_sm_out = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(conv_out_sm)
        bg_sm_out = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(conv_out_sm)

        pz_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 0], name='pzu')(
            unsupervised_label)
        cz_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 1], name='czu')(
            unsupervised_label)
        us_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 2], name='usu')(
            unsupervised_label)
        afs_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 3], name='afsu')(
            unsupervised_label)
        bg_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 4], name='bgu')(
            unsupervised_label)

        pz = K.stack([pz_ensemble_pred, supervised_flag])
        cz = K.stack([cz_ensemble_pred, supervised_flag])
        us = K.stack([us_ensemble_pred, supervised_flag])
        afs = K.stack([afs_ensemble_pred, supervised_flag])
        bg = K.stack([bg_ensemble_pred, supervised_flag])

        if trained_model is not None:
            p_model_normal = Model([input_img, unsupervised_label, supervised_flag],
                                   [pz_sm_out, cz_sm_out, us_sm_out, afs_sm_out, bg_sm_out])

            p_model_normal.load_weights(trained_model, by_name=True)
            # load weights
            weights_list = p_model_normal.get_weights()

        #################################end###########################

        #####################MC########################################

        conv1, conv1_b_m = self.downLayer_MC(input_img, sfs, 1, bn)
        conv2, conv2_b_m = self.downLayer_MC(conv1, sfs * 2, 2, bn)

        conv3 = Conv3D(sfs * 4, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv_mc' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv_mc' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        # conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)

        conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_1_mc')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4 = Dropout(0.5, seed=4, name='Dropout_mc' + str(4))(conv4, training=True)
        conv4 = Conv3D(sfs * 16, (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv4_2_mc')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        # conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv3DTranspose(sfs * 16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same',
                              name='up_mc' + str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv_mc' + str(5) + '_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed=5, name='Dropout_mc' + str(5))(conv5, training=True)
        conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                       name='conv_mc' + str(5) + '_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer_MC(conv5, conv2_b_m, sfs * 8, 6, bn, do)
        conv7 = self.upLayer_MC(conv6, conv1_b_m, sfs * 4, 7, bn, do)

        conv_out_sm = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax_mc')(conv7)

        pz_sm_out_mc = Lambda(lambda x: x[:, :, :, :, 0], name='pz_mc')(conv_out_sm)
        cz_sm_out_mc = Lambda(lambda x: x[:, :, :, :, 1], name='cz_mc')(conv_out_sm)
        us_sm_out_mc = Lambda(lambda x: x[:, :, :, :, 2], name='us_mc')(conv_out_sm)
        afs_sm_out_mc = Lambda(lambda x: x[:, :, :, :, 3], name='afs_mc')(conv_out_sm)
        bg_sm_out_mc = Lambda(lambda x: x[:, :, :, :, 4], name='bg_mc')(conv_out_sm)

        p_model_MC = Model([input_img, unsupervised_label, supervised_flag],
                           [pz_sm_out_mc, cz_sm_out_mc, us_sm_out_mc, afs_sm_out_mc, bg_sm_out_mc])

        p_model_MC.set_weights(weights_list)
        for layer in p_model_MC.layers:
            layer.trainable = False

        ########################################################
        # merged_model = Model([input_img, unsupervised_label, supervised_flag],
        #                      [pz_sm_out, cz_sm_out, us_sm_out, afs_sm_out, bg_sm_out, pz_sm_out_mc, cz_sm_out_mc,
        #                       us_sm_out_mc, afs_sm_out_mc, bg_sm_out_mc])

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        if (nb_gpus is None):
            p_model_normal.compile(optimizer=optimizer,
                                   loss={'pz': self.semi_supervised_loss(pz, unsup_loss_class_wt=1),
                                         'cz': self.semi_supervised_loss(cz, 1),
                                         'us': self.semi_supervised_loss(us, 1),
                                         'afs': self.semi_supervised_loss(afs, 1),
                                         'bg': self.semi_supervised_loss(bg, 1)
                                         }
                                   ,
                                   metrics={'pz': [self.dice_coef, self.unsup_dice_tb(pz, 1), self.dice_tb(pz, 1)],
                                            'cz': [self.dice_coef, self.unsup_dice_tb(cz, 1), self.dice_tb(cz, 1)],
                                            'us': [self.dice_coef, self.unsup_dice_tb(us, 1), self.dice_tb(us, 1)],
                                            'afs': [self.dice_coef, self.unsup_dice_tb(afs, 1), self.dice_tb(afs, 1)],
                                            'bg': [self.dice_coef, self.unsup_dice_tb(bg, 1), self.dice_tb(bg, 1)]
                                            },
                                   loss_weights={'pz': 1, 'cz': 1, 'us': 1, 'afs': 1, 'bg': 1}
                                   )
        else:
            with tf.device(gpu_id):
                p_model_normal = multi_gpu_model(p_model_normal, gpus=nb_gpus)

                p_model_normal.compile(optimizer=optimizer,
                                       loss={'pz': self.semi_supervised_loss(pz, unsup_loss_class_wt=1),
                                             'cz': self.semi_supervised_loss(cz, 1),
                                             'us': self.semi_supervised_loss(us, 1),
                                             'afs': self.semi_supervised_loss(afs, 1),
                                             'bg': self.semi_supervised_loss(bg, 1)
                                             }
                                       ,
                                       metrics={'pz': [self.dice_coef, self.unsup_dice_tb(pz, 1), self.dice_tb(pz, 1)],
                                                'cz': [self.dice_coef, self.unsup_dice_tb(cz, 1), self.dice_tb(cz, 1)],
                                                'us': [self.dice_coef, self.unsup_dice_tb(us, 1),
                                                       self.dice_tb(us, 1)],
                                                'afs': [self.dice_coef, self.unsup_dice_tb(afs, 1),
                                                        self.dice_tb(afs, 1)],
                                                'bg': [self.dice_coef, self.unsup_dice_tb(bg, 1), self.dice_tb(bg, 1)]
                                                },
                                       loss_weights={'pz': 1, 'cz': 1, 'us': 1, 'afs': 1, 'bg': 1}
                                       )

        return p_model_MC, p_model_normal
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg, input_idx])
