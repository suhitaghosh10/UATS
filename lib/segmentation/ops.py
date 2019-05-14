import numpy as np
from keras import backend as K
from keras.losses import mean_squared_error


def ramp_up_weight(ramp_period, weight_max):
    """Ramp-Up weight generator.
    The function is used in unsupervised component of loss.
    Returned weight ramps up until epoch reaches ramp_period
    """
    cur_epoch = 0

    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-5 * (1 - T) ** 2) * weight_max
        else:
            yield 1 * weight_max

        cur_epoch += 1


def ramp_down_weight(ramp_period):
    """Ramp-Down weight generator"""
    cur_epoch = 1

    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-12.5 * T ** 2)
        else:
            yield 0

        cur_epoch += 1


def dice_coef(y_true, y_pred, smooth=1., axis=(-3, -2, -1)):
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    # along each image
    intersection = K.sum(y_true * y_pred, axis=axis)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return (2. * intersection + smooth) / (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth)


def semi_supervised_loss(num_class):
    """custom loss function"""
    epsilon = 1e-08
    smooth = 1.

    def loss_func(y_true, y_pred):
        """semi-supervised loss function
        the order of y_true:
        unsupervised_target(num_class), supervised_label(num_class), supervised_flag(1), unsupervised weight(1)
        """
        #y_true -> 1st part is unsuper label(0->9) since num_cl=10, 2nd part is supervised label (10->19), 20th is flag, 21st/last one is weight
        unsupervised_target = y_true[:, :, :, :, 0]
        print('unsupervised_target', K.int_shape(unsupervised_target))
        supervised_label = y_true[:, :, :, :, 1]
        print('supervised_target', K.int_shape(supervised_label))
        supervised_flag = y_true[:, :, :, :, 2]
        print('supervised_flag', K.int_shape(supervised_flag))
        weight = y_true[:, :, :, :, -1]  # last elem are weights
        print('wt', K.int_shape(weight), 'value', weight)

        model_pred = y_pred[:, :, :, :, 0]
        print(K.int_shape(y_pred))


        # weighted unsupervised loss over batchsize
        unsupervised_loss = weight * K.mean(mean_squared_error(unsupervised_target, model_pred))
        print('unsupervised_loss', unsupervised_loss)

        # To sum over only supervised data on categorical_crossentropy, supervised_flag(1/0) is used

        supervised_loss = - K.mean(
            K.sum(supervised_label * K.log(K.clip(model_pred, epsilon, 1.0 - epsilon)), axis=1) * supervised_flag)

        supervised_loss = - K.mean(dice_coef(supervised_label, model_pred) * supervised_flag[:, 0, 0, 0])

        return supervised_loss + unsupervised_loss

    return loss_func


def update_weight(y, unsupervised_weight, next_weight):
    """update weight of the unsupervised part of loss"""
    y[:, :, :, :, -1] = next_weight
    unsupervised_weight[:] = next_weight

    return y, unsupervised_weight


def update_unsupervised_target(ensemble_prediction, y, num_class, alpha, cur_pred, epoch):
    """update ensemble_prediction and unsupervised weight when an epoch ends"""
    # Z = αZ + (1 - α)z
    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred

    # initial_epoch = 0
    y[:, :, :, :, 0:num_class] = ensemble_prediction / (1 - alpha ** (epoch + 1))

    return ensemble_prediction, y


def evaluate(model, num_class, num_test, test_x, test_y):
    """evaluate"""
    test_supervised_label_dummy = np.zeros((num_test, 32, 168, 168, num_class))
    test_supervised_flag_dummy = np.zeros((num_test, 32, 168, 168, 1))
    test_unsupervised_weight_dummy = np.zeros((num_test, 32, 168, 168, num_class))

    test_x_ap = [test_x, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy]
    print(test_y.shape, test_supervised_label_dummy.shape, test_supervised_flag_dummy.shape,
          test_unsupervised_weight_dummy.shape)
    test_y_ap = np.concatenate(
        [test_y, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy], axis=-1)
    p = model.predict(x=test_x_ap, batch_size=1, verbose=1)
    print(p.shape)
    # test_x_c = np.concatenate((test_x, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy), axis=-1)
    # print(model.evaluate(test_x_ap, test_y_ap, batch_size=1, verbose=1))
    pr = p[: ,:, :, :, 0:num_class]
    pr_arg_max = np.argmax(pr, axis=-1)
    tr_arg_max = np.argmax(test_y, axis=-1)
    cnt = np.sum(pr_arg_max == tr_arg_max) / (num_test * 32 * 168 * 168)
    print('Test Accuracy: ', cnt, flush=True)
    axis = (-4, -3, -2, -1)
    intersection = np.sum(test_y * pr, axis=axis)
    # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice = (2. * intersection + 1.) / (np.sum(test_y, axis=axis) + np.sum(pr, axis=axis) + 1.)
    print('Test DSC: ', np.mean(dice), flush=True)
