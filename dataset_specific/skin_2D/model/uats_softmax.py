import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, \
    Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from utility.weight_norm import AdamWithWeightnorm
# from lib.segmentation.group_norm import GroupNormalization

smooth = 1.
consistency_wt = 1


class weighted_model:
    alpha = 0.6
    epoch_ctr = K.variable(0, name='epoch_ctr')

    class LossCallback(Callback):

        def on_epoch_end(self, epoch, logs={}):
            weighted_model.epoch_ctr = epoch
            print(weighted_model.epoch_ctr)

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

    def semi_supervised_loss(self, input, alpha=0.6):

        def loss_func(y_true, y_pred):

            supervised_flag = input[1, :, :, :]
            val_flag = False
            if supervised_flag[0, 0, 0] == 3:
                print('validation')
                val_flag = True

            unsupervised_gt = input[0, :, :, :]
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

            return supervised_loss + consistency_wt * unsupervised_loss

        return loss_func

    def c_dice_loss(self, y_true, y_pred, weight, smooth=1., axis=(1, 2)):

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

    def unsup_dice_tb(self, input):

        def unsup_dice_loss(y_true, y_pred, smooth=1., axis=(1, 2), alpha=0.6):
            supervised_flag = input[1, :, :, :]
            val_flag = False

            unsupervised_gt = input[0, :, :, :]
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

            return consistency_wt * unsupervised_loss

        return unsup_dice_loss

    def dice_tb(self, input, class_wt=1., ):
        def dice_loss(y_true, y_pred, smooth=1., axis=(1, 2)):
            supervised_flag = input[1, :, :, :]
            unsupervised_gt = input[0, :, :, :]
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

    def unsup_c_dice_loss(self, y_true, y_pred, smooth=1., axis=(1, 2)):

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

    def build_model(self, img_shape=(32, 168, 168), learning_rate=5e-5, gpu_id=None,
                    nb_gpus=None,
                    trained_model=None,
                    temp=1):

        input_img = Input(img_shape, name='img_inp')
        unsupervised_label = Input((img_shape[0], img_shape[1], 2), name='unsup_label_inp')
        supervised_flag = Input((img_shape[0], img_shape[1]), name='flag_inp')

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
        bg_out = Lambda(lambda x: x[:, :, :, 0], name='bg')(conv_out)
        skin_out = Lambda(lambda x: x[:, :, :, 1], name='skin')(conv_out)

        bg_ensemble_pred = Lambda(lambda x: x[:, :, :, 0], name='bgu')(
            unsupervised_label)
        skin_ensemble_pred = Lambda(lambda x: x[:, :, :, 1], name='skinu')(
            unsupervised_label)
        bg = K.stack([bg_ensemble_pred, supervised_flag])
        skin = K.stack([skin_ensemble_pred, supervised_flag])

        optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        #optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        if (nb_gpus is None):
            p_model = Model([input_img, unsupervised_label, supervised_flag],
                            [bg_out, skin_out])
            if trained_model is not None:
                p_model.load_weights(trained_model)
            p_model.compile(optimizer=optimizer,
                            loss={'bg': self.semi_supervised_loss(bg),
                                  'skin': self.semi_supervised_loss(skin)},
                            metrics={
                                'bg': [self.dice_coef, self.unsup_dice_tb(bg), self.dice_tb(bg)],
                                'skin': [self.dice_coef, self.unsup_dice_tb(skin), self.dice_tb(skin)],

                            }
                            )
        else:
            with tf.device(gpu_id):
                model = Model([input_img, unsupervised_label, supervised_flag],
                              [conv_out])
                if trained_model is not None:
                    model.load_weights(trained_model)

                p_model = multi_gpu_model(model, gpus=nb_gpus)
                p_model.compile(optimizer=optimizer,
                                loss={'bg': self.semi_supervised_loss(bg, unsup_loss_class_wt=1),
                                      'skin': self.semi_supervised_loss(skin, 1)},
                                metrics={
                                    'bg': [self.dice_coef, self.unsup_dice_tb(bg, 1), self.dice_tb(bg, 1)],
                                    'skin': [self.dice_coef, self.unsup_dice_tb(skin, 1), self.dice_tb(skin, 1)],

                                }
                                )

        return p_model
