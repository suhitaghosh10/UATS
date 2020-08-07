import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Dropout
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

    def dice_tb(self, input, class_wt=1.):
        def dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            supervised_flag = input[:, :, :, :, 1]
            y_true = y_true * supervised_flag
            y_pred = y_pred * supervised_flag

            intersection = K.sum(y_true * y_pred, axis=axis)
            y_true_sum = K.sum(y_true, axis=axis)
            y_pred_sum = K.sum(y_pred, axis=axis)

            avg_dice_coef = K.mean((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth))

            return - avg_dice_coef * class_wt

        return dice_loss

    def unsup_dice_tb(self, input, class_wt=1.):
        def unsup_dice_loss(y_true, y_pred, smooth=1., axis=(1, 2, 3)):
            y_true = y_true
            y_pred = y_pred

            intersection = K.sum(y_true * y_pred, axis=axis)
            y_true_sum = K.sum(y_true, axis=axis)
            y_pred_sum = K.sum(y_pred, axis=axis)

            avg_dice_coef = K.mean((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth))

            return 1 - avg_dice_coef

            avg_dice_coef = K.mean((2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth))

            return (1 - avg_dice_coef) * class_wt

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

    def focal_loss(self, y_true, y_pred, weight, gamma=1.5, axis=(1, 2, 3)):

        y_true = y_true
        y_pred = y_pred

        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = K.clip(pt, K.epsilon(), 1 - K.epsilon())
        CE = -K.log(pt)
        FL = K.pow(1 - pt, gamma) * CE
        class_pixel_count = tf.to_float(K.sum(y_true))
        total_count = tf.to_float(tf.size(y_true))
        class_ratio = total_count / class_pixel_count

        # weight_non_zero_count = tf.add(tf.to_float(K.sum(weight, axis=axis)), K.epsilon())
        avg_FL = K.mean(FL) * class_ratio

        return - avg_FL

    def focal_tb(self, input, unsup_loss_wt=1.):
        def focal_loss(y_true, y_pred, gamma=1.5, alpha=0.6, axis=(1, 2, 3)):
            unsupervised_gt = input[:, :, :, :, 0]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            weight = input[:, :, :, :, 2]  # last elem are weights

            y_true = unsupervised_gt * weight
            y_pred = y_pred * weight

            pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
            pt = K.clip(pt, K.epsilon(), 1 - K.epsilon())
            CE = -K.log(pt)
            FL = K.pow(1 - pt, gamma) * CE
            class_pixel_count = tf.to_float(K.sum(y_true))
            total_count = tf.to_float(tf.size(y_true))
            class_ratio = total_count / class_pixel_count

            # weight_non_zero_count = tf.add(tf.to_float(K.sum(weight, axis=axis)), K.epsilon())
            avg_FL = K.mean(FL) * class_ratio * unsup_loss_wt

            return - avg_FL

        return focal_loss

    def semi_supervised_loss(self, input, unsup_loss_class_wt=1., alpha=0.6):

        def loss_func(y_true, y_pred):
            print(K.eval(self.epoch_ctr))

            unsupervised_gt = input[:, :, :, :, 0]
            unsupervised_gt = unsupervised_gt / (1 - alpha ** (self.epoch_ctr + 1))
            unsupervised_gt = tf.where(K.greater(unsupervised_gt, K.constant(0.9)), K.ones_like(unsupervised_gt),
                                       K.zeros_like(unsupervised_gt))
            supervised_flag = input[:, :, :, :, 1]
            weight = input[:, :, :, :, 2]  # last elem are weights

            supervised_loss = self.dice_loss(y_true, y_pred, supervised_flag)
            # unsupervised_loss = self.focal_loss(unsupervised_gt, y_pred, weight)
            unsupervised_loss = self.unsup_dice_loss(unsupervised_gt, y_pred, weight)

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

    def build_model(self, img_shape=(32, 168, 168), num_class=5, learning_rate=5e-5, gpu_id=None,
                    nb_gpus=None,
                    trained_model=None):
        input_img = Input((*img_shape, 1), name='img_inp')
        unsupervised_label = Input((*img_shape, 5), name='unsup_label_inp')
        supervised_flag = Input(shape=(*img_shape, 1), name='flag_inp')

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

        conv_out = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv_final_softmax')(conv7)

        afs = concatenate([unsupervised_label, supervised_flag], name='afs_c')

        # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

        if (nb_gpus is None):
            p_model = Model([input_img, unsupervised_label, supervised_flag], [afs])
            if trained_model is not None:
                p_model.load_weights(trained_model)

            # model_copy = Model([input_img, unsupervised_label, supervised_flag, unsupervised_weight],[pz_out, cz_out, us_out, afs_out, bg_out])

            # intermediate_layer_model = Model(inputs=training_scripts.input,outputs=training_scripts.get_layer(layer_name).output)

            p_model.compile(optimizer=optimizer,
                            loss={'afs': self.semi_supervised_loss(afs, 2)},
                            metrics={'afs': [self.dice_coef, self.unsup_dice_tb(afs, 1), self.dice_tb(afs, 1)]}
                            )
        else:
            with tf.device(gpu_id):
                model = Model([input_img, unsupervised_label, supervised_flag], [afs])
                if trained_model is not None:
                    model.load_weights(trained_model)

                p_model = multi_gpu_model(model, gpus=nb_gpus)

                # model_copy = Model([input_img, unsupervised_label, gt, supervised_flag, unsupervised_weight],[pz_out, cz_out, us_out, afs_out, bg_out])

                # intermediate_layer_model = Model(inputs=training_scripts.input,outputs=training_scripts.get_layer(layer_name).output)

                p_model.compile(optimizer=optimizer,
                                loss={'afs': self.semi_supervised_loss(afs, 2)},
                                metrics={'afs': [self.dice_coef, self.unsup_dice_tb(afs, 1), self.dice_tb(afs, 1)]}
                                )

        return p_model
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg])
    # return Model([input_img, supervised_label, supervised_flag, unsupervised_weight], [pz, cz, us, afs, bg, input_idx])
