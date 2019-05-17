from keras import backend as K

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger

from generator.data_gen import DataGenerator
from lib.segmentation.ops import ramp_up_weight, semi_supervised_loss, ramp_down_weight, dice_coef
from lib.segmentation.utils import split_supervised_train, make_train_test_dataset
from lib.segmentation.weight_norm import AdamWithWeightnorm
from zonal_utils.AugmentationGenerator import *


def main():
    # 50,000 Training -> 4000 supervised data(400 per class) and 46,000 unsupervised data.
    # Prepare args
    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    num_labeled_train = 38
    num_test = 28
    ramp_up_period = 80
    ramp_down_period = 50
    num_class = 5
    num_epoch = 351
    batch_size = 2
    weight_max = 30
    learning_rate = 5e-5
    alpha = 0.6
    weight_norm_flag = True
    augmentation_flag = False
    whitening_flag = False
    trans_range = 2
    EarlyStop = False
    LRScheduling = True

    # Data Preparation
    train_x = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
    train_y = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy')
    num_train_data = train_x.shape[0]

    val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy')

    # datagen list
    train_id_list = [str(i) for i in np.arange(0, train_x.shape[0])]
    val_id_list = [str(i) for i in np.arange(0, val_x.shape[0])]

    # prepare weights and arrays for updates
    gen_weight = ramp_up_weight(ramp_up_period, weight_max * (num_labeled_train / num_train_data))
    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # idx_list = [v for v in range(num_train_data)]

    class TemporalCallback(Callback):
        def __init__(self, train_idx_list, unsupervised_target, supervised_label, supervised_flag, unsupervised_weight):
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            self.unsupervised_target = unsupervised_target
            self.supervised_label = supervised_label
            self.supervised_flag = supervised_flag
            self.unsupervised_weight = unsupervised_weight

            # initial epoch
            self.ensemble_prediction = np.zeros((num_train_data, 32, 168, 168, num_class))
            self.cur_pred = K.zeros((num_train_data, 32, 168, 168, num_class))

        def on_batch_begin(self, batch, logs=None):
            start = batch * batch_size
            self.batch_idx_list = self.train_idx_list[start: start + batch_size]


        def on_batch_end(self, batch, logs=None):
            i = batch * batch_size
            idx_list = np.int_(self.train_idx_list[i:i + batch_size])
            for i, idx in enumerate(idx_list):
                out = K.stack((model.outputs[0][:, :, :, :, 0],
                               model.outputs[1][:, :, :, :, 0],
                               model.outputs[2][:, :, :, :, 0],
                               model.outputs[3][:, :, :, :, 0],
                               model.outputs[4][:, :, :, :, 0]), axis=-1)

                self.cur_pred[idx].assign(out)





        def on_epoch_begin(self, epoch, logs=None):
            # shuffle examples
            np.random.shuffle(self.train_idx_list)
            np.random.shuffle(val_id_list)

            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)

        def on_epoch_end(self, epoch, logs={}):
            next_weight = next(gen_weight)
            # update ensemble_prediction and unsupervised weight when an epoch ends
            unsupervised_weight[:, :, :, :, :] = next_weight
            # Z = αZ + (1 - α)z
            self.ensemble_prediction = alpha * self.ensemble_prediction + (1 - alpha) * self.cur_pred
            self.unsupervised_target = self.ensemble_prediction / (1 - alpha ** (epoch + 1))

            DataGenerator.__init__(self, train_x, unsupervised_target, supervised_label, supervised_flag,
                                   unsupervised_weight,
                                   self.train_idx_list)

        def get_training_list(self):
            return self.train_idx_list

    ret_dic = split_supervised_train(train_x, train_y, num_labeled_train)

    ret_dic['test_x'] = val_x
    ret_dic['test_y'] = val_y
    ret_dic = make_train_test_dataset(ret_dic, num_class)

    unsupervised_target = ret_dic['unsupervised_target']
    supervised_label = ret_dic['supervised_label']
    supervised_flag = ret_dic['train_sup_flag']
    unsupervised_weight = ret_dic['unsupervised_weight']

    # make the whole data and labels for training
    # y = np.concatenate((unsupervised_target, supervised_label, supervised_flag, unsupervised_weight), axis=-1)

    # Build Model
    from lib.segmentation.model_WN import build_model
    # optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model = build_model(num_class=num_class, learning_rate=learning_rate)

    optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                  loss=semi_supervised_loss(),
                  metrics={'pz': dice_coef, 'cz': dice_coef, 'us': dice_coef,
                           'afs': dice_coef, 'bg': dice_coef}
                  )

    # model.metrics_tensors += model.outputs
    model.summary()

    # callback
    csv_logger = CSVLogger('validation.csv', append=True, separator=';')
    model_checkpoint = ModelCheckpoint('./temporal.h5', monitor='val_loss', save_best_only=True, verbose=1,
                                       mode='min')
    tensorboard = TensorBoard(log_dir='./tensor_temporal', write_graph=False, write_grads=True, histogram_freq=0,
                              batch_size=5,
                              write_images=False)
    earlyStopImprovement = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, verbose=1, mode='min')
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)

    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    print("Images Size:", num_train_data)
    print("GT Size:", train_y.shape)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    tcb = TemporalCallback(train_id_list, unsupervised_target, supervised_label, supervised_flag, unsupervised_weight)
    cb = [csv_logger, model_checkpoint, tcb]
    if EarlyStop:
        cb.append(earlyStopImprovement)
    if LRScheduling:
        cb.append(LRDecay)
    cb.append(tensorboard)

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size,
              'n_classes': 5,
              'n_channels': 1,
              'shuffle': True}

    training_generator = DataGenerator(train_x, unsupervised_target, supervised_label, supervised_flag,
                                       unsupervised_weight, tcb.get_training_list(), **params)

    steps = num_train_data / 2

    val_unsupervised_target = np.zeros((val_x.shape[0], 32, 168, 168, num_class))
    val_supervised_flag = np.ones((val_x.shape[0], 32, 168, 168, 1))
    val_unsupervised_weight = np.zeros((val_x.shape[0], 32, 168, 168, 5))

    pz = np.stack((val_unsupervised_target[:, :, :, :, 0], val_y[:, :, :, :, 0],
                   val_supervised_flag[:, :, :, :, 0], val_unsupervised_weight[:, :, :, :, 0]),
                  axis=-1)

    cz = np.stack((val_unsupervised_target[:, :, :, :, 1], val_y[:, :, :, :, 1],
                   val_supervised_flag[:, :, :, :, 0], val_unsupervised_weight[:, :, :, :, 1]),
                  axis=-1)

    us = np.stack((val_unsupervised_target[:, :, :, :, 2], val_y[:, :, :, :, 2],
                   val_supervised_flag[:, :, :, :, 0], val_unsupervised_weight[:, :, :, :, 2]),
                  axis=-1)

    afs = np.stack((val_unsupervised_target[:, :, :, :, 3], val_y[:, :, :, :, 3],
                    val_supervised_flag[:, :, :, :, 0], val_unsupervised_weight[:, :, :, :, 3]),
                   axis=-1)

    bg = np.stack((val_unsupervised_target[:, :, :, :, 4], val_y[:, :, :, :, 0],
                   val_supervised_flag[:, :, :, :, 0], val_unsupervised_weight[:, :, :, :, 4]),
                  axis=-1)

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x, val_y, val_supervised_flag, val_unsupervised_weight]

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  use_multiprocessing=False,
                                  epochs=num_epoch,
                                  callbacks=cb)
    # workers=4)
    model.save('temporal_final.h5')

    # Training

    # for epoch in range(num_epoch):
    # print('epoch: ', epoch)
    # idx_list = shuffle(idx_list)

    # if epoch > num_epoch - ramp_down_period:
    #   weight_down = next(gen_lr_weight)
    #   K.set_value(model.optimizer.lr, weight_down * learning_rate)
    #   K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)

    # ave_loss = 0
    # for i in range(0, num_train_data, batch_size):
    #  target_idx = idx_list[i:i + batch_size]
    # done umntil here

    # if augmentation_flag:
    #    x1 = data_augmentation_tempen(train_x[target_idx], trans_range)
    # else:
    #    x1 = train_x[target_idx]

    # x2 = supervised_label[target_idx]
    # x3 = supervised_flag[target_idx]
    # x4 = unsupervised_weight[target_idx]
    # y_t = y[target_idx]
    # mess starts here
    # predicted_index = 0
    # label_index = 5
    # flag_index = 10
    # wt_index = 11
    # pz = np.stack((y[target_idx, :, :, :, predicted_index], y[target_idx, :, :, :, label_index],
    #               y[target_idx, :, :, :, flag_index], y[target_idx, :, :, :, wt_index]), axis=-1)
    # cz = np.stack((y[target_idx, :, :, :, predicted_index + 1], y[target_idx, :, :, :, label_index + 1],
    #               y[target_idx, :, :, :, flag_index], y[target_idx, :, :, :, wt_index + 1]), axis=-1)
    # us = np.stack((y[target_idx, :, :, :, predicted_index + 2], y[target_idx, :, :, :, label_index + 2],
    #               y[target_idx, :, :, :, flag_index], y[target_idx, :, :, :, wt_index + 2]), axis=-1)
    # afs = np.stack((y[target_idx, :, :, :, predicted_index + 3], y[target_idx, :, :, :, label_index + 3],
    #                y[target_idx, :, :, :, flag_index], y[target_idx, :, :, :, wt_index + 3]), axis=-1)
    # bg = np.stack((y[target_idx, :, :, :, predicted_index + 4], y[target_idx, :, :, :, label_index + 4],
    #               y[target_idx, :, :, :, flag_index], y[target_idx, :, :, :, wt_index + 4]), axis=-1)

    # y_t = [pz, cz, us, afs, bg]
    # x_t = [x1, x2, x3, x4]

    # loss, pz_loss, cz_loss, us_loss, afs_loss, bg_loss, output_0, output_1, output_2, output_3, output_4 = model.train_on_batch(
    #    x=x_t, y=y_t)

    # cur_pred[idx_list[i:i + batch_size], :, :, :, 0] = output_0[:, :, :, :, 0]
    # cur_pred[idx_list[i:i + batch_size], :, :, :, 1] = output_1[:, :, :, :, 0]
    # cur_pred[idx_list[i:i + batch_size], :, :, :, 2] = output_2[:, :, :, :, 0]
    # cur_pred[idx_list[i:i + batch_size], :, :, :, 3] = output_3[:, :, :, :, 0]
    # cur_pred[idx_list[i:i + batch_size], :, :, :, 4] = output_4[:, :, :, :, 0]

    # ave_loss += loss

    # print('Training Loss: ', (ave_loss * batch_size) / (num_train_data), flush=True)

    # Update phase
    # next_weight = next(gen_weight)
    # y, unsupervised_weight = update_weight(y, unsupervised_weight, next_weight)
    # ensemble_prediction, y = update_unsupervised_target(ensemble_prediction, y, num_class, alpha, cur_pred, epoch)

    # Evaluation
    # if epoch % 5 == 0:
    # print('Evaluate epoch :  ', epoch, flush=True)
    #evaluate(model, num_class, 20, test_x, test_y)


if __name__ == '__main__':
    main()
