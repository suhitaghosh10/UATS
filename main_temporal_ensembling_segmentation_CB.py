from keras import backend as K

import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger

from generator.data_gen import *
from generator.val_data_gen import ValDataGenerator
from lib.segmentation.ops import ramp_up_weight, semi_supervised_loss, ramp_down_weight
from lib.segmentation.utils import split_supervised_train, make_train_test_dataset
from lib.segmentation.weight_norm import AdamWithWeightnorm
from zonal_utils.AugmentationGenerator import *


class MainClass:
    # 50,000 Training -> 4000 supervised data(400 per class) and 46,000 unsupervised data.
    # Prepare args
    # args = parse_args()

    num_labeled_train = 38
    num_test = 20
    ramp_up_period = 80
    ramp_down_period = 50
    num_train_data = 58
    num_class = 5
    num_epoch = 351
    bs = 2
    weight_max = 30
    learning_rate = 0.001
    alpha = 0.6
    weight_norm_flag = True
    augmentation_flag = False
    whitening_flag = False
    trans_range = 2
    EarlyStop = False
    LRScheduling = True

    train_id_list = []
    val_id_list = []
    cur_pred = np.zeros((num_labeled_train + num_test, 32, 168, 168, num_class))
    ensemble_prediction = np.zeros((num_labeled_train + num_test, 32, 168, 168, num_class))

    class NewCallback(Callback):
        def __init__(self, bs, ramp_down_period, model, learning_rate):
            self.ramp_down_period = ramp_down_period
            self.model = model
            self.bs = bs
            self.learning_rate = learning_rate

        def ramp_down_weight(self, ramp_period):
            """Ramp-Down weight generator"""
            cur_epoch = 1

            while True:
                if cur_epoch <= ramp_period - 1:
                    T = (1 / (ramp_period - 1)) * cur_epoch
                    yield np.exp(-12.5 * T ** 2)
                else:
                    yield 0

                cur_epoch += 1

        def on_batch_end(self, batch, logs=None):
            i = self.bs * batch

            print('gandu', i)
            MainClass.cur_pred[MainClass.train_id_list[i:i + self.bs], :, :, :, 0]

        # cur_pred[idx_list[i:i + batch_size], :, :, :, 1] = output_1[:, :, :, :, 0]
        # cur_pred[idx_list[i:i + batch_size], :, :, :, 2] = output_2[:, :, :, :, 0]
        # cur_pred[idx_list[i:i + batch_size], :, :, :, 3] = output_3[:, :, :, :, 0]
        # cur_pred[idx_list[i:i + batch_size], :, :, :, 4] = output_4[:, :, :, :, 0]

        def on_epoch_begin(self, epoch, logs=None):
            if K.greater(epoch, self.epochs_num - self.ramp_down_period):
                weight_down = next(self.ramp_down_weight(self.ramp_down_period))
                K.set_value(self.model.optimizer.lr, weight_down * self.learning_rate)
                K.set_value(self.model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('alpha-----', K.get_value(self.model.optimizer.lr))

        def on_epoch_end(self, epoch, logs={}):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                next_weight = next(self.ramp_down_weight(self.ramp_down_period))
                # do this in data generator
                ensemble_prediction = MainClass.alpha * MainClass.ensemble_prediction + (
                            1 - MainClass.alpha) * MainClass.cur_pred

                # initial_epoch = 0
                MainClass.ensemble_prediction = ensemble_prediction / (1 - MainClass.alpha ** (epoch + 1))

                # self.ensemble_prediction, y = update_unsupervised_target(self.ensemble_prediction, y, num_class, alpha, cur_pred,epoch)

    def get_next_weight(self):
        return next(ramp_up_weight(MainClass.ramp_up_period,
                                   MainClass.weight_max * (MainClass.num_labeled_train / MainClass.num_train_data)))

    def get_updated_prediction(self):
        return MainClass.ensemble_prediction

    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        # Data Preparation
        train_x = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
        train_y = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy')

        val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
        val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy')

        ret_dic = split_supervised_train(train_x, train_y, MainClass.num_labeled_train)

        ret_dic['test_x'] = val_x
        ret_dic['test_y'] = val_y
        ret_dic = make_train_test_dataset(ret_dic, MainClass.num_class)

        unsupervised_target = ret_dic['unsupervised_target']
        supervised_label = ret_dic['supervised_label']
        supervised_flag = ret_dic['train_sup_flag']
        unsupervised_weight = ret_dic['unsupervised_weight']
        # test = ret_dic['test_y']

        # make the whole data and labels for training
        # y = np.concatenate((unsupervised_target, supervised_label, supervised_flag, unsupervised_weight), axis=-1)

        num_train_data = train_x.shape[0]

        from lib.segmentation.model_WN import build_model
        optimizer = AdamWithWeightnorm(lr=MainClass.learning_rate, beta_1=0.9, beta_2=0.999)

        model = build_model(num_class=5)
        model.compile(optimizer=optimizer,
                      loss=semi_supervised_loss(5))

        model.metrics_tensors += model.outputs
        model.summary()

        # prepare weights and arrays for updates
        gen_weight = ramp_up_weight(MainClass.ramp_up_period,
                                    MainClass.weight_max * (MainClass.num_labeled_train / num_train_data))
        gen_lr_weight = ramp_down_weight(MainClass.ramp_down_period)

        # ensemble_prediction = np.zeros((num_train_data, 32, 168, 168, num_class))
        # cur_pred = np.zeros((num_train_data, 32, 168, 168, num_class))

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

        cb = [csv_logger, model_checkpoint]
        if MainClass.EarlyStop:
            cb.append(earlyStopImprovement)
        if MainClass.LRScheduling:
            cb.append(LRDecay)
        cb.append(tensorboard)

        print('BATCH Size = ', 2)

        print('Callbacks: ', cb)
        params = {'dim': (32, 168, 168),
                  'batch_size': MainClass.bs,
                  'n_classes': 5,
                  'n_channels': 1,
                  'shuffle': True}

        train_id_list = [str(i) for i in np.arange(0, train_x.shape[0])]
        training_generator = DataGenerator(train_x, unsupervised_target, supervised_label, supervised_flag,
                                           unsupervised_weight, train_id_list, **params)

        val_id_list = [str(i) for i in np.arange(0, val_x.shape[0])]
        val_generator = ValDataGenerator(val_x, val_id_list, **params)

        steps = num_train_data / 2

        history = model.fit_generator(generator=training_generator,
                                      steps_per_epoch=steps,
                                      validation_data=val_generator,
                                      use_multiprocessing=False,
                                      epochs=MainClass.num_epoch,
                                      callbacks=cb)
        # workers=4)
        model.save('temporal_final.h5')


if __name__ == '__main__':
    m = MainClass()
    m.train()
