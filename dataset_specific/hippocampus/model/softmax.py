import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


# from lib.segmentation.weight_norm import AdamWithWeightnorm


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

    def dice_tb(self, input, class_wt=1., ):
        def dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            supervised_flag = input[1, :, :, :, :]
            unsupervised_gt = input[0, :, :, :, :]
            alpha = 0.6
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
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

        def unsup_dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3), alpha=0.6):
            supervised_flag = input[1, :, :, :, :]
            val_flag = False

            unsupervised_gt = input[0, :, :, :, :]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            # for confident voxels (denoted by 2) change from 0 to 1 (unlabelled to labelled), but keep the background(0) as 0
            y_true_final = tf.where(tf.equal(supervised_flag, 2), unsupervised_gt, y_true)
            # reset flag
            supervised_flag = tf.where(tf.equal(supervised_flag, 2), K.ones_like(supervised_flag), supervised_flag)
            # check its validation data, then no consistency loss
            # validation data (denoted by 3)
            supervised_flag = tf.where(tf.equal(supervised_flag, 3), K.ones_like(supervised_flag), supervised_flag)
            supervised_loss = self.c_dice_loss(y_true_final, y_pred, supervised_flag)

            # for validation loss, make unsup_loss_class_wt zero
            if val_flag:
                return 0
            else:
                intersection = K.sum(y_true * y_pred, axis=axis)
                y_true_sum = K.sum(y_true, axis=axis)
                y_pred_sum = K.sum(y_pred, axis=axis)

                sign_pred = tf.where(tf.greater(y_pred, 0), K.ones_like(y_pred), K.zeros_like(y_pred))
                if tf.greater(intersection, 0) is not None:
                    c = K.sum(y_true * y_pred) / (K.sum(y_true * sign_pred) + K.epsilon())
                else:
                    c = 1

            avg_dice_coef = K.mean((2. * intersection + smooth) / ((c * y_pred_sum) + y_true_sum + smooth))
            temp = 1 - avg_dice_coef
            unsupervised_loss = tf.where(tf.greater(K.abs(temp), K.abs(supervised_loss)), K.zeros_like(temp), temp)

            return unsupervised_loss

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

        # unsup_loss_class_wt = unsup_loss_class_wt

        def loss_func(y_true, y_pred, unsup_loss_class_wt=unsup_loss_class_wt):
            # print(K.eval(self.epoch_ctr))

            supervised_flag = input[1, :, :, :, :]
            val_flag = False
            if supervised_flag[0, 0, 0, 0] is 3:
                val_flag = True

            unsupervised_gt = input[0, :, :, :, :]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            # for confident voxels (denoted by 2) change from 0 to 1 (unlabelled to labelled), but keep the background(0) as 0
            y_true_final = tf.where(tf.equal(supervised_flag, 2), unsupervised_gt, y_true)
            # reset flag
            supervised_flag = tf.where(tf.equal(supervised_flag, 2), K.ones_like(supervised_flag), supervised_flag)
            # check its validation data, then no consistency loss
            # validation data (denoted by 3)
            supervised_flag = tf.where(tf.equal(supervised_flag, 3), K.ones_like(supervised_flag), supervised_flag)
            supervised_loss = self.c_dice_loss(y_true_final, y_pred, supervised_flag)

            # for validation loss, make unsup_loss_class_wt zero
            if val_flag:
                unsupervised_loss = 0
            else:
                temp = self.unsup_c_dice_loss(unsupervised_gt, y_pred)
                unsupervised_loss = tf.where(tf.greater(K.abs(temp), K.abs(supervised_loss)), K.zeros_like(temp), temp)

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

    def build_model(self, img_shape=(32, 168, 168), learning_rate=5e-5, gpu_id=None,
                    nb_gpus=None,
                    trained_model=None):
        input_img = Input((*img_shape, 1), name='img_inp')
        unsupervised_label = Input((*img_shape, 3), name='unsup_label_inp')
        supervised_flag = Input(shape=img_shape, name='flag_inp')

        kernel_init = 'he_normal'
        sfs = 8  # start filter size
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

        conv_out = Conv3D(3, (1, 1, 1), name='conv_final_softmax', activation='softmax')(conv7)

        bg_sm_out = Lambda(lambda x: x[:, :, :, :, 0], name='bg')(conv_out)
        z1_sm_out = Lambda(lambda x: x[:, :, :, :, 1], name='z1')(conv_out)
        z2_sm_out = Lambda(lambda x: x[:, :, :, :, 2], name='z2')(conv_out)

        bg_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 0], name='bgu')(
            unsupervised_label)
        z1_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 1], name='z1u')(
            unsupervised_label)
        z2_ensemble_pred = Lambda(lambda x: x[:, :, :, :, 2], name='z2u')(
            unsupervised_label)

        bg = K.stack([bg_ensemble_pred, supervised_flag])
        z1 = K.stack([z1_ensemble_pred, supervised_flag])
        z2 = K.stack([z2_ensemble_pred, supervised_flag])

        # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        if (nb_gpus is None):
            p_model = Model([input_img, unsupervised_label, supervised_flag],
                            [bg_sm_out, z1_sm_out, z2_sm_out])
            if trained_model is not None:
                p_model.load_weights(trained_model)

            p_model.compile(optimizer=optimizer,
                            loss={'bg': self.semi_supervised_loss(bg, unsup_loss_class_wt=1),
                                  'z1': self.semi_supervised_loss(z1, 1),
                                  'z2': self.semi_supervised_loss(z2, 1),
                                  }
                            ,
                            metrics={'bg': [self.dice_coef, self.unsup_dice_tb(bg, 1), self.dice_tb(bg, 1)],
                                     'z1': [self.dice_coef, self.unsup_dice_tb(z1, 1), self.dice_tb(z1, 1)],
                                     'z2': [self.dice_coef, self.unsup_dice_tb(z2, 1), self.dice_tb(z2, 1)],
                                     },
                            loss_weights={'bg': 1, 'z1': 1, 'z2': 1}
                            )
        else:
            with tf.device(gpu_id):
                model = Model([input_img, unsupervised_label, supervised_flag],
                              [bg_sm_out, z1_sm_out, z2_sm_out])
                if trained_model is not None:
                    model.load_weights(trained_model)

                p_model = multi_gpu_model(model, gpus=nb_gpus)

                p_model.compile(optimizer=optimizer,
                                loss={'bg': self.semi_supervised_loss(bg, unsup_loss_class_wt=1),
                                      'z1': self.semi_supervised_loss(z1, 1),
                                      'z2': self.semi_supervised_loss(z2, 1)
                                      }
                                ,
                                metrics={'bg': [self.dice_coef, self.unsup_dice_tb(bg, 1), self.dice_tb(bg, 1)],
                                         'z1': [self.dice_coef, self.unsup_dice_tb(z1, 1), self.dice_tb(z1, 1)],
                                         'z2': [self.dice_coef, self.unsup_dice_tb(z2, 1), self.dice_tb(z2, 1)]
                                         },
                                loss_weights={'bg': 1, 'z1': 1, 'z2': 1}
                                )

        return p_model
