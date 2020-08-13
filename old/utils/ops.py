import numpy as np

ZONE = {0: 'pz', 1: 'cz', 2: 'us', 3: 'afs', 4: 'bg'}


def ramp_up_weight(ramp_period, weight_max, cur_epoch):
    """Ramp-Up weight generator.
    The function is used in unsupervised component of loss.
    Returned weight ramps up until epoch reaches ramp_period
    """

    while True:
        if cur_epoch <= ramp_period - 1:
            T = (1 / (ramp_period - 1)) * cur_epoch
            yield np.exp(-5 * (1 - T) ** 2) * weight_max
        else:
            yield 1 * weight_max


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
    test_supervised_flag_dummy = np.ones((num_test, 32, 168, 168, 1))
    test_unsupervised_weight_dummy = np.zeros((num_test, 32, 168, 168, num_class))

    # test_x_ap = [test_x, test_supervised_label_dummy, test_supervised_flag_dummy, test_unsupervised_weight_dummy]
    test_x_ap = [test_x, test_unsupervised_weight_dummy, test_y, test_supervised_flag_dummy,
                 test_unsupervised_weight_dummy]
    print(test_y.shape, test_supervised_label_dummy.shape, test_supervised_flag_dummy.shape,
          test_unsupervised_weight_dummy.shape)
    test_y_ap = np.concatenate(
        [test_supervised_label_dummy, test_y, test_supervised_flag_dummy, test_unsupervised_weight_dummy], axis=-1)
    p = model.predict(x=test_x_ap, batch_size=1, verbose=1)
    # predicted = np.zeros((num_test, 32, 168, 168, num_class))
    # for index in np.arange(0,5):
    #   predicted[:, :, :, :, index] = p[index]
    dice = np.zeros(5)
    for index in np.arange(0, 5):
        pr = p[index][:, :, :, :]
        gt = test_y[:, :, :, :, index]
        # pr_arg_max = np.argmax(pr, axis=-1)
        # tr_arg_max = np.argmax(gt, axis=-1)
        # cnt = np.sum(pr_arg_max == tr_arg_max) / (num_test * 32 * 168 * 168)
        # print('Validation Accuracy: ', cnt, flush=True)
        axis = (-3, -2, -1)
        intersection = np.sum(gt * pr, axis=axis)
        # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice[index] = np.mean((2. * intersection + 1.) / (np.sum(gt, axis=axis) + np.sum(pr, axis=axis) + 1.))

    print('Validation DSC: ', ZONE[0], ': ', dice[0], ZONE[1], ': ', dice[1], ZONE[2], ': ', dice[2], ZONE[3], ': ',
          dice[3], ZONE[4], ': ', dice[4], flush=True)
