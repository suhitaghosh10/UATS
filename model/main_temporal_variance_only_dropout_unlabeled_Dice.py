import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard

from generator.data_gen_optim import DataGenerator
from lib.segmentation.model_WN_MCdropout import build_model
from lib.segmentation.ops import ramp_down_weight, ramp_up_weight
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import make_dataset
from zonal_utils.AugmentationGenerator import *

# 58 Training 236 unsupervised data.

TB_LOG_DIR = './tb/variance_mcdropout/mse'
MODEL_NAME = './temporal_mse.h5'
NUM_CLASS = 5
num_epoch = 351
batch_size = 2

# hyper-params
UPDATE_WTS_AFTER_EPOCH = 1
ENSEMBLE_NO = 3
ramp_up_period = 100
ramp_down_period = 100
# weight_max = 40
weight_max = 1
learning_rate = 5e-5
alpha = 0.6
VAR_THRESHOLD = 0.5


def train(gpu_id, nb_gpus):

        # train
        # change here
        train_x_arr = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
        train_y_arr = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy').astype('int8')
        train_ux_arr = np.load('/home/suhita/zonals/data/training/trainArray_unlabeled_imgs_fold1.npy')
        train_ux_predicted_arr = np.load('/home/suhita/zonals/data/training/trainArray_unlabeled_GT_fold1.npy')

        val_x_arr = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
        val_y_arr = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

        num_un_labeled_train = train_ux_arr.shape[0]
        num_labeled_train = train_x_arr.shape[0]
        num_train_data = num_un_labeled_train + num_labeled_train

        ret_dic = make_dataset(train_x_arr, train_y_arr, train_ux_arr, train_ux_predicted_arr, val_x_arr,
                               val_y_arr, 5)
        del train_x_arr, train_y_arr, train_ux_arr, train_ux_predicted_arr

        imgs = ret_dic['train_x']
        unsupervised_target = ret_dic['unsupervised_target']
        # supervised_label = ret_dic['supervised_label']
        supervised_flag = ret_dic['train_sup_flag']
        unsupervised_weight = ret_dic['unsupervised_weight']

        del ret_dic

        # datagen listmake_dataset
        train_id_list = [str(i) for i in np.arange(0, num_train_data)]
        val_id_list = [str(i) for i in np.arange(0, val_x_arr.shape[0])]

        # prepare weights and arrays for updates
        gen_weight = ramp_up_weight(ramp_up_period, weight_max)
        gen_lr_weight = ramp_down_weight(ramp_down_period)

        # prepare dataset
        print('-' * 30)
        print('Loading train data...')
        print('-' * 30)

        # ret_dic = split_supervised_train(train_x, train_y, num_labeled_train)
        # Build Model
        trained_model_path = '/home/suhita/zonals/data/model.h5'
        model = build_model(num_class=NUM_CLASS, use_dice_cl=True, learning_rate=learning_rate, gpu_id=gpu_id,
                            nb_gpus=nb_gpus, trained_model=trained_model_path)

        print("Images Size:", num_train_data)
        print("Unlabeled Size:", num_un_labeled_train)

        print('-' * 30)
        print('Creating and compiling model...')
        print('-' * 30)

        # model.metrics_tensors += model.outputs
        model.summary()

        class TemporalCallback(Callback):

            def __init__(self, imgs, unsupervised_target, unsupervised_weight, train_idx_list):
                self.imgs = imgs
                self.ensemble_prediction = unsupervised_target
                self.unsupervised_weight = unsupervised_weight
                self.supervised_flag = supervised_flag
                self.train_idx_list = train_idx_list  # list of indexes of training eg

            def on_batch_begin(self, batch, logs=None):
                pass

            def on_epoch_begin(self, epoch, logs=None):

                if epoch > num_epoch - ramp_down_period:
                    weight_down = next(gen_lr_weight)
                    K.set_value(model.optimizer.lr, weight_down * learning_rate)
                    K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                    print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))

            def on_epoch_end(self, epoch, logs={}):

                if epoch >= UPDATE_WTS_AFTER_EPOCH - 5:
                    # update unsupervised weight one step b4 update weights for prev
                    inp = [self.imgs, self.ensemble_prediction, self.supervised_flag, self.unsupervised_weight]

                    cur_pred = np.zeros((num_train_data, 32, 168, 168, NUM_CLASS))

                    model_out = model.predict(inp, batch_size=2, verbose=1)  # 1
                    model_out = np.add(model_out, model.predict(inp, batch_size=2, verbose=1))  # 2
                    model_out = np.add(model_out, model.predict(inp, batch_size=2, verbose=1))  # 3
                    del inp

                    cur_pred[:, :, :, :, 0] = model_out[0] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 1] = model_out[1] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 2] = model_out[2] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 3] = model_out[3] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 4] = model_out[4] / ENSEMBLE_NO
                    del model_out

                    # Z = αZ + (1 - α)z
                    self.ensemble_prediction = alpha * self.ensemble_prediction + (1 - alpha) * cur_pred
                    # self.unsupervisedprediction = self.ensemble_prediction / (1 - alpha ** (epoch + 1))

                if epoch >= UPDATE_WTS_AFTER_EPOCH:
                    # update unsupervised weight
                    next_weight = next(gen_weight)
                    # self.unsupervised_weight = (1. - np.abs(cur_pred - self.ensemble_prediction)) * next_weight
                    self.unsupervised_weight = np.where(np.abs(cur_pred - self.ensemble_prediction) >= VAR_THRESHOLD,
                                                        0., 1.) * next_weight

                if 'cur_pred' in locals(): del cur_pred

                # shuffle examples
                np.random.shuffle(self.train_idx_list)
                np.random.shuffle(val_id_list)
                DataGenerator('/home/suhita/zonals/data/training/',
                              self.ensemble_prediction,
                              self.unsupervised_weight,
                              self.supervised_flag,
                              self.train_idx_list)

            def get_training_list(self):
                return self.train_idx_list

            def get_ensemble_prediction(self):
                return self.ensemble_prediction

            def get_unsupervised_weight(self):
                return self.unsupervised_weight

            def get_supervised_flag(self):
                return self.supervised_flag

        # callbacks
        print('-' * 30)
        print('Creating callbacks...')
        print('-' * 30)
        # csv_logger = CSVLogger('validation.csv', append=True, separator=';')
        # model_checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True,verbose=1, mode='min')
        if nb_gpus > 1:
            model_checkpoint = ModelCheckpointParallel(MODEL_NAME,
                                                       monitor='val_loss',
                                                       save_best_only=True,
                                                       verbose=1,
                                                       mode='min')
        else:
            model_checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss',
                                               save_best_only=True,
                                               verbose=1,
                                               mode='min')

        tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=True, histogram_freq=0,
                                  batch_size=5, write_images=False)

        tcb = TemporalCallback(imgs, unsupervised_target, unsupervised_weight, train_id_list)
        del unsupervised_target, unsupervised_weight, supervised_flag, imgs
        cb = [model_checkpoint, tcb]
        cb.append(tensorboard)

        print('BATCH Size = ', batch_size)

        print('Callbacks: ', cb)
        params = {'dim': (32, 168, 168),
                  'batch_size': batch_size}

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        training_generator = DataGenerator('/home/suhita/zonals/data/training/',
                                           tcb.get_ensemble_prediction(),
                                           tcb.get_unsupervised_weight(),
                                           tcb.get_supervised_flag(),
                                           tcb.get_training_list(),
                                           **params)

        steps = num_train_data / batch_size

        val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168, 1), dtype='int8')
        val_unsupervised_weight = np.zeros((val_x_arr.shape[0], 32, 168, 168, 5), dtype='int8')

        pz = val_y_arr[:, :, :, :, 0]
        cz = val_y_arr[:, :, :, :, 1]
        us = val_y_arr[:, :, :, :, 2]
        afs = val_y_arr[:, :, :, :, 3]
        bg = val_y_arr[:, :, :, :, 4]

        y_val = [pz, cz, us, afs, bg]
        x_val = [val_x_arr, val_y_arr, val_supervised_flag, val_unsupervised_weight]
        del val_supervised_flag, val_unsupervised_weight, pz, cz, us, afs, bg, val_y_arr, val_x_arr
        history = model.fit_generator(generator=training_generator,
                                      steps_per_epoch=steps,
                                      validation_data=[x_val, y_val],
                                      epochs=num_epoch,
                                      callbacks=cb
                                      )

        # workers=4)
        # model.save('temporal_max_ramp_final.h5')


def predict(val_x_arr, val_y_arr):
    val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168, 1), dtype='int8')
    val_unsupervised_weight = np.zeros((val_x_arr.shape[0], 32, 168, 168, 5), dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr, val_y_arr, val_supervised_flag, val_unsupervised_weight]

    model = build_model(num_class=NUM_CLASS, use_dice_cl=False, learning_rate=learning_rate, gpu_id=None,
                        nb_gpus=None, trained_model='/home/suhita/zonals/temporal/temporal_dice.h5')
    print('load_weights')
    #model.load_weights()
    print('predict')
    #out = model.predict(x_val, batch_size=1, verbose=1)

    print(model.evaluate(x_val, y_val, batch_size=1, verbose=1))

        #np.save(name + '.npy', out)


if __name__ == '__main__':
    gpu = '/CPU:0'
    batch_size = 2
    gpu_id = '2, 3'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train(gpu, nb_gpus)
    # val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

    # val_x = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_imgs.npy')
    # val_y = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')
    #predict(val_x, val_y)
