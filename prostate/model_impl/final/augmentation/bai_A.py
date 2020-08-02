from time import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from generator.temporal_A import DataGenerator as train_gen
from lib.segmentation.model.bai import weighted_model
from lib.segmentation.ops import ramp_down_weight
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_array, save_array
from zonal_utils.AugmentationGenerator import *
from shutil import copyfile
from kits.utils import makedir
import shutil
import keras.backend as K

learning_rate = 5e-5
AUGMENTATION_NO = 1
augmentation = True
FOLD_NUM = 2
NR_CLASS = 5
num_epoch = 1000
batch_size = 2
UPDATE_EPOCH_NO = 50
DIM = [32, 168, 168]

ramp_up_period = 50
ramp_down_period = 50
alpha = 0.6

ENS_GT_PATH = '/data/suhita/temporal/prostate/output/sadv223/'


def train(gpu_id, nb_gpus, perc):
    PERCENTAGE_OF_LABELLED = perc
    DATA_PATH = '/cache/suhita/data/prostate/fold_' + str(FOLD_NUM) + '_P' + str(PERCENTAGE_OF_LABELLED) + '/train/'
    TRAIN_NUM = 58
    NAME = 'prostate_bai_F' + str(FOLD_NUM) + '_Perct_Labelled_' + str(PERCENTAGE_OF_LABELLED)
    TB_LOG_DIR = '/data/suhita/temporal/tb/prostate/' + NAME + '_' + str(learning_rate) + '/'
    MODEL_NAME = '/data/suhita/temporal/prostate/' + NAME + '.h5'
    TRAINED_MODEL_PATH = '/data/suhita/prostate/supervised_F' + str(FOLD_NUM) + '_P' + str(perc) + '.h5'
    CSV_NAME = '/data/suhita/temporal/CSV/' + NAME + '.csv'
    num_labeled_train = int(PERCENTAGE_OF_LABELLED * TRAIN_NUM)
    num_train_data = len(os.listdir(DATA_PATH + '/imgs/'))
    num_un_labeled_train = num_train_data - num_labeled_train

    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model = wm.build_model(TRAINED_MODEL_PATH, img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate,
                           gpu_id=gpu_id,
                           nb_gpus=nb_gpus)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, data_path, ensemble_path, train_idx_list):

            self.data_path = data_path
            self.ensemble_path = ensemble_path
            self.train_idx_list = train_idx_list

            if not os.path.exists(self.ensemble_path):
                makedir(self.ensemble_path)
                makedir(os.path.join(self.ensemble_path, 'ens_gt'))
                makedir(os.path.join(self.ensemble_path, 'flag'))
            for patient in np.arange(num_train_data):
                copyfile(os.path.join(DATA_PATH, 'GT', str(patient) + '.npy'),
                         os.path.join(self.ensemble_path, 'ens_gt', str(patient) + '.npy'))

                np.save(os.path.join(self.ensemble_path, 'flag', str(patient) + '.npy'),
                        np.ones(shape=DIM, dtype='float16'))

        def on_batch_begin(self, batch, logs=None):
            pass

        def shall_save(self, cur_val, prev_val):
            flag_save = False
            val_save = prev_val

            if cur_val > prev_val:
                flag_save = True
                val_save = cur_val

            return flag_save, val_save

        def on_epoch_begin(self, epoch, logs=None):

            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))
            # print(K.eval(model.layers[43].trainable_weights[0]))

        def on_epoch_end(self, epoch, logs={}):
            if epoch > 0 and epoch % UPDATE_EPOCH_NO == 0:

                patients_per_batch = 59
                num_batches = num_un_labeled_train // patients_per_batch
                remainder = num_un_labeled_train % patients_per_batch
                num_batches = num_batches + 1 if remainder > 0 else num_batches

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if (
                            b_no < num_batches) else remainder
                    start = (b_no * patients_per_batch) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array(self.data_path + '/imgs/', start, end)
                    ensemble_prediction = get_array(self.ensemble_path + '/ens_gt/', start, end, dtype='float32')
                    supervised_flag = get_array(self.ensemble_path + '/flag/', start, end, dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NR_CLASS))
                    # cur_sigmoid_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))
                    model_out = model.predict(inp, batch_size=2, verbose=1)  # 1

                    # model_out = np.add(model_out, model_impl.predict(inp, batch_size=2, verbose=1))  # 2
                    del inp

                    cur_pred[:, :, :, :, 0] = model_out[0]
                    cur_pred[:, :, :, :, 1] = model_out[1]
                    cur_pred[:, :, :, :, 2] = model_out[2]
                    cur_pred[:, :, :, :, 3] = model_out[3]
                    cur_pred[:, :, :, :, 4] = model_out[4]

                    del model_out

                    save_array(os.path.join(self.ensemble_path, 'ens_gt'), cur_pred, start, end)
                    del ensemble_prediction, cur_pred

    # callbacks
    print('-' * 30)
    print('Creating callbacks...')
    print('-' * 30)
    csv_logger = CSVLogger(CSV_NAME, append=True, separator=';')
    # model_checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True,verbose=1, mode='min')
    if nb_gpus is not None and nb_gpus > 1:
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
                              batch_size=1, write_images=False)

    train_id_list = np.arange(num_train_data)
    np.random.shuffle(train_id_list)

    print(train_id_list[0:10])

    np.random.shuffle(train_id_list)
    tcb = TemporalCallback(DATA_PATH, ENS_GT_PATH, train_id_list)
    lcb = wm.LossCallback()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.001)
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger, es]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    # params = {'dim': (32, 168, 168),'batch_size': batch_size}

    print('-' * 30)
    print('Fitting model_impl...')
    print('-' * 30)
    training_generator = train_gen(DATA_PATH,
                                   ENS_GT_PATH,
                                   train_id_list,
                                   batch_size=batch_size,
                                   labelled_num=num_labeled_train)

    # steps = num_train_data / batch_size
    steps = (num_train_data * AUGMENTATION_NO) / batch_size
    # steps = 2

    val_fold = os.listdir(DATA_PATH[:-7] + '/val/imgs/')
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data, DIM[0], DIM[1], DIM[2]), dtype='int8')
    val_img_arr = np.zeros((num_val_data, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((num_val_data, DIM[0], DIM[1], DIM[2], NR_CLASS), dtype=float)

    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(DATA_PATH[:-7] + '/val/imgs/' + str(i) + '.npy')
        val_GT_arr[i] = np.load(DATA_PATH[:-7] + '/val/gt/' + str(i) + '.npy')

    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr[:, :, :, :, 0], val_GT_arr[:, :, :, :, 1],
             val_GT_arr[:, :, :, :, 2], val_GT_arr[:, :, :, :, 3],
             val_GT_arr[:, :, :, :, 4]]
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )
    del model

    # workers=4)
    # model_impl.save('temporal_max_ramp_final.h5')


def cleanup():
    shutil.rmtree(ENS_GT_PATH)
    K.clear_session()


if __name__ == '__main__':
    batch_size = batch_size

    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # train(gpu, nb_gpus)
    try:
        # train(None, None, 0.1)
        # cleanup()
        train(None, None, 1.0)
        # cleanup()
        # train(None, None, 0.5)
        # cleanup()
        # train(None, None, 1.0)

    finally:

        if os.path.exists(ENS_GT_PATH):
            cleanup()
        print('clean up done!')

    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
