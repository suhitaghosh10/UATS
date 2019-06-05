import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.models import Model
from keras.utils import multi_gpu_model

# from keras.optimizers import Adam
from lib.segmentation.weight_norm import AdamWithWeightnorm


class weighted_model:
    epoch_ctr = K.variable(0, name='epoch_ctr')

    class LossCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):
            weighted_model.epoch_ctr = epoch
            print(weighted_model.epoch_ctr)

    def dice_coef(self, y_true, y_pred, smooth=1., axis=(-3, -2, -1)):
        # intersection = K.sum(y_true * y_pred, axis=axis)
        # return (2. * intersection + smooth) / (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def c_dice_coef(self, y_true, y_pred, smooth=1.):
        # y_true_f = K.flatten(y_true)
        # y_pred_f = K.flatten(y_pred)
        size_of_A_intersect_B = K.sum(y_true * y_pred)
        size_of_A = K.sum(y_true)
        size_of_B = K.sum(y_pred)
        sign_B = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
        if tf.greater(size_of_A_intersect_B, 0) is not None:
            c = K.sum(y_true * y_pred) / K.sum(y_true * sign_B)
        else:
            c = 1

        return ((2. * size_of_A_intersect_B) + smooth) / ((c * size_of_A) + size_of_B + smooth)

    def semi_supervised_loss_mse(self, input, alpha=0.6):

        def loss_func(y_true, y_pred):
            print(K.eval(self.epoch_ctr))
            """semi-supervised loss function
                the order of y_true:
                unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised weight(1)
            """
            unsupervised_gt = input[:, :, :, :, 0]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            supervised_flag = input[:, :, :, :, 1]
            weight = input[:, :, :, :, 2]  # last elem are weights

            # model_pred = y_pred

            unsupervised_loss = - K.mean(weight * K.square(unsupervised_gt - y_pred))
            # print('unsupervised_loss', unsupervised_loss)

            supervised_loss = - K.mean(self.dice_coef(y_true, y_pred) * supervised_flag[:, 0, 0, 0])

            return supervised_loss + unsupervised_loss

        return loss_func

    def semi_supervised_loss_dice(self, input, alpha=0.6):
        """custom loss function"""
        epsilon = 1e-08
        smooth = 1.

        def loss_func(y_true, y_pred):
            """semi-supervised loss function
            the order of y_true:
            unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised weight(1)
            """
            unsupervised_gt = input[:, :, :, :, 0]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            supervised_flag = input[:, :, :, :, 1]
            weight = input[:, :, :, :, 2]  # last elem are weights

            unsupervised_loss = - K.mean(weight * self.c_dice_coef(unsupervised_gt, y_pred))
            supervised_loss = - K.mean(self.dice_coef(y_true, y_pred) * supervised_flag[:, 0, 0, 0])

            return supervised_loss + unsupervised_loss

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
            conv = Dropout(0.5, seed=3, name='Dropout_' + str(i))(conv, training=True)
        conv = Conv3D(int(filterSize / 2), (3, 3, 3), activation='relu', padding='same', name='conv' + str(i) + '_2')(
            conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv

    def build_model(self, img_shape=(32, 168, 168), use_dice_cl=None, num_class=5, learning_rate=5e-5, gpu_id=None,
                    nb_gpus=None,
                trained_model=None):
        input_img = Input((*img_shape, 1), name='img_inp')
        unsupervised_label = Input((*img_shape, 5), name='unsup_label_inp')
        supervised_flag = Input(shape=(*img_shape, 1), name='flag_inp')
        unsupervised_weight = Input(shape=(*img_shape, num_class), name='wt_inp')

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
            conv4 = Dropout(0.5, seed=4, name='Dropout_' + str(4))(conv4, training=True)
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
            conv5 = Dropout(0.5, seed=5, name='Dropout_' + str(5))(conv5, training=True)
        conv5 = Conv3D(int(sfs * 8), (3, 3, 3), activation='relu', padding='same', kernel_initializer=kernel_init,
                   name='conv' + str(5) + '_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer(conv5, conv2_b_m, sfs * 8, 6, bn, do)
        conv7 = self.upLayer(conv6, conv1_b_m, sfs * 4, 7, bn, do)

        conv_out = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        pz_sm_out = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(conv_out)
        cz_sm_out = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(conv_out)
        us_sm_out = Lambda(lambda x: x[:, :, :, :, 2], name='us')(conv_out)
        afs_sm_out = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(conv_out)
        bg_sm_out = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(conv_out)

        pz_ensemble_pred = Lambda(lambda x: K.reshape(x[:, :, :, :, 0], tf.convert_to_tensor([-1, *img_shape, 1])),
                              name='pzu')(
        unsupervised_label)
        cz_ensemble_pred = Lambda(lambda x: K.reshape(x[:, :, :, :, 1], tf.convert_to_tensor([-1, *img_shape, 1])),
                              name='czu')(
        unsupervised_label)
        us_ensemble_pred = Lambda(lambda x: K.reshape(x[:, :, :, :, 2], tf.convert_to_tensor([-1, *img_shape, 1])),
                              name='usu')(
        unsupervised_label)
        afs_ensemble_pred = Lambda(lambda x: K.reshape(x[:, :, :, :, 3], tf.convert_to_tensor([-1, *img_shape, 1])),
                               name='afsu')(
        unsupervised_label)
        bg_ensemble_pred = Lambda(lambda x: K.reshape(x[:, :, :, :, 4], tf.convert_to_tensor([-1, *img_shape, 1])),
                              name='bgu')(
        unsupervised_label)

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

        pz = concatenate([pz_ensemble_pred, supervised_flag, pz_wt], name='pz_c')
        cz = concatenate([cz_ensemble_pred, supervised_flag, cz_wt], name='cz_c')
        us = concatenate([us_ensemble_pred, supervised_flag, us_wt], name='us_c')
        afs = concatenate([afs_ensemble_pred, supervised_flag, afs_wt], name='afs_c')
        bg = concatenate([bg_ensemble_pred, supervised_flag, bg_wt], name='bg_c')

    # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        if (nb_gpus is None):
            p_model = Model([input_img, unsupervised_label, supervised_flag, unsupervised_weight],
                        [pz_sm_out, cz_sm_out, us_sm_out, afs_sm_out, bg_sm_out])
            if trained_model is not None:
                p_model.load_weights(trained_model)

        # model_copy = Model([input_img, unsupervised_label, supervised_flag, unsupervised_weight],[pz_out, cz_out, us_out, afs_out, bg_out])

        # intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

            if use_dice_cl:
                p_model.compile(optimizer=optimizer,
                                loss={'pz': self.semi_supervised_loss_dice(pz),
                                      'cz': self.semi_supervised_loss_dice(cz),
                                      'us': self.semi_supervised_loss_dice(us),
                                      'afs': self.semi_supervised_loss_dice(afs),
                                      'bg': self.semi_supervised_loss_dice(bg)},
                                metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                         'afs': self.dice_coef, 'bg': self.dice_coef}
                            )
            else:
                p_model.compile(optimizer=optimizer,
                                loss={'pz': self.semi_supervised_loss_mse(pz), 'cz': self.semi_supervised_loss_mse(cz),
                                      'us': self.semi_supervised_loss_mse(us),
                                      'afs': self.semi_supervised_loss_mse(afs),
                                      'bg': self.semi_supervised_loss_mse(bg)},
                                metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                         'afs': self.dice_coef, 'bg': self.dice_coef}
                            )
        else:
            with tf.device(gpu_id):
                model = Model([input_img, unsupervised_label, supervised_flag, unsupervised_weight],
                          [pz_sm_out, cz_sm_out, us_sm_out, afs_sm_out, bg_sm_out])
                if trained_model is not None:
                    model.load_weights(trained_model)

                p_model = multi_gpu_model(model, gpus=nb_gpus)

            #model_copy = Model([input_img, unsupervised_label, gt, supervised_flag, unsupervised_weight],[pz_out, cz_out, us_out, afs_out, bg_out])

        # intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

                if use_dice_cl:
                    p_model.compile(optimizer=optimizer,
                                    loss={'pz': self.semi_supervised_loss_dice(pz),
                                          'cz': self.semi_supervised_loss_dice(cz),
                                          'us': self.semi_supervised_loss_dice(us),
                                          'afs': self.semi_supervised_loss_dice(afs),
                                          'bg': self.semi_supervised_loss_dice(bg)},
                                    metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                             'afs': self.dice_coef, 'bg': self.dice_coef}
                                )
                else:
                    p_model.compile(optimizer=optimizer,
                                    loss={'pz': self.semi_supervised_loss_mse(pz),
                                          'cz': self.semi_supervised_loss_mse(cz),
                                          'us': self.semi_supervised_loss_mse(us),
                                          'afs': self.semi_supervised_loss_mse(afs),
                                          'bg': self.semi_supervised_loss_mse(bg)},
                                    metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                                             'afs': self.dice_coef, 'bg': self.dice_coef}
                                )

        return p_model
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg, input_idx])
