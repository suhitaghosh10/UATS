import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard

from generator.data_gen_optim import DataGenerator
from lib.segmentation.model_WN_MCdropout import weighted_model
from lib.segmentation.ops import ramp_down_weight
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_complete_array, get_array, save_array
from zonal_utils.AugmentationGenerator import *

# 294 Training 58 have gt
learning_rate = 5e-6
TB_LOG_DIR = '/home/suhita/zonals/temporal/tb/variance_mcdropout/sl2_v2' + str(learning_rate) + '/'
MODEL_NAME = '/home/suhita/zonals/temporal/temporal_sl2_v2.h5'

TRAIN_IMGS_PATH = '/home/suhita/zonals/data/training/imgs/'
TRAIN_GT_PATH = '/home/suhita/zonals/data/training/gt/'
# TRAIN_UNLABELED_DATA_PRED_PATH = '/home/suhita/zonals/data/training/ul_gt/'

VAL_IMGS_PATH = '/home/suhita/zonals/data/test_anneke/imgs/'
VAL_GT_PATH = '/home/suhita/zonals/data/test_anneke/gt/'

# TRAINED_MODEL_PATH = '/home/suhita/zonals/data/model.h5'
TRAINED_MODEL_PATH = '/home/suhita/zonals/temporal/temporal_s2.h5'

WEIGHT_PATH = '/home/suhita/zonals/temporal/sad/wt/'
ENS_GT_PATH = '/home/suhita/zonals/temporal/sad/ens_gt/'

NUM_CLASS = 5
num_epoch = 351
batch_size = 2
IMGS_PER_ENS_BATCH = 100

# hyper-params
# UPDATE_WTS_AFTER_EPOCH = 1
ENSEMBLE_NO = 1
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 50
# weight_max = 40
weight_max = 30

alpha = 0.6
VAR_THRESHOLD = 0.5


def train(gpu_id, nb_gpus):
    num_labeled_train = 58
    num_train_data = len(os.listdir(TRAIN_IMGS_PATH))
    num_un_labeled_train = num_train_data - num_labeled_train
    num_val_data = len(os.listdir(VAL_IMGS_PATH))

    supervised_flag = np.concatenate(
        (np.ones((num_labeled_train, 32, 168, 168, 1)), np.zeros((num_un_labeled_train, 32, 168, 168, 1)))).astype(
        'int8')

    # prepare weights and arrays for updates
    # gen_weight = ramp_up_weight(ramp_up_period, weight_max * (num_labeled_train / num_train_data))
    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model

    wm = weighted_model()
    model = wm.build_model(num_class=NUM_CLASS, use_dice_cl=False, learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    # model.metrics_tensors += model.outputs
    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, imgs_path, gt_path, ensemble_path, weight_path, supervised_flag,
                     variance_threshold, train_idx_list):
            self.imgs_path = imgs_path
            self.gt_path = gt_path
            self.ensemble_path = ensemble_path
            self.weight_path = weight_path
            self.supervised_flag = supervised_flag.astype('int8')
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            self.variance_th = variance_threshold

            unsupervised_target = get_complete_array(TRAIN_GT_PATH, dtype='float32')

            for patient in np.arange(num_train_data):
                np.save(self.weight_path + str(patient) + '.npy', np.ones((32, 168, 168, NUM_CLASS)).astype('float32'))
                np.save(self.ensemble_path + str(patient) + '.npy', unsupervised_target[patient])
            del unsupervised_target

        def on_batch_begin(self, batch, logs=None):
            pass

        def on_epoch_begin(self, epoch, logs=None):

            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))

        def on_epoch_end(self, epoch, logs={}):

            # if epoch >= ramp_up_period - 5:
            # if epoch >= SAVE_WTS_AFTR_EPOCH:
            # next_weight = next(gen_weight)
            #   print('rampup wt', next_weight)
            sup_no = 0.
            if epoch <= 100:
                THRESHOLD = 0.9
            else:
                THRESHOLD = 0.99

            if epoch >= 0:
                # update unsupervised weight one step b4 update weights for prev
                patients_per_batch = IMGS_PER_ENS_BATCH
                num_batches = num_train_data // patients_per_batch
                remainder = num_train_data % patients_per_batch
                num_batches = num_batches if remainder is 0 else num_batches + 1

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if b_no < num_batches - 1 else remainder
                    start = b_no * patients_per_batch
                    end = (start + actual_batch_size)
                    imgs = get_array(self.imgs_path, start, end)
                    ensemble_prediction = get_array(self.ensemble_path, start, end)
                    wt = get_array(self.weight_path, start, end, data_type='float32')

                    inp = [imgs, ensemble_prediction, self.supervised_flag[start:end], wt]
                    del imgs, wt

                    cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))

                    model_out = model.predict(inp, batch_size=2, verbose=1)  # 1
                    # model_out = np.add(model_out, model.predict(inp, batch_size=2, verbose=1))  # 2
                    # model_out = np.add(model_out, model.predict(inp, batch_size=2, verbose=1))  # 3
                    del inp

                    cur_pred[:, :, :, :, 0] = model_out[0] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 1] = model_out[1] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 2] = model_out[2] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 3] = model_out[3] / ENSEMBLE_NO
                    cur_pred[:, :, :, :, 4] = model_out[4] / ENSEMBLE_NO

                    del model_out

                    # Z = αZ + (1 - α)z
                    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred
                    # del cur_pred
                    save_array(self.ensemble_path, ensemble_prediction, start, end)
                    # del ensemble_prediction
                    if b_no == 0:
                        flag = self.supervised_flag[num_labeled_train:actual_batch_size]
                        # flag = np.where(np.reshape(np.max(cur_pred[num_labeled_train:actual_batch_size], axis=-1), flag.shape)  >= 0.99, np.ones_like(flag), flag)
                        flag = np.where(
                            np.reshape(np.max(ensemble_prediction[num_labeled_train:actual_batch_size], axis=-1),
                                       flag.shape) >= THRESHOLD, np.ones_like(flag), np.zeros_like(flag))
                        self.supervised_flag[num_labeled_train:actual_batch_size] = flag
                        del flag
                    else:
                        flag = self.supervised_flag[start:end]
                        # flag = np.where(np.reshape(np.max(cur_pred, axis=-1),flag.shape) >= 0.99, np.ones_like(flag), flag)
                        flag = np.where(np.reshape(np.max(ensemble_prediction, axis=-1), flag.shape) >= THRESHOLD,
                                        np.ones_like(flag), np.zeros_like(flag))
                        self.supervised_flag[start:end] = flag
                        del flag
                    sup_no = sup_no + np.count_nonzero(self.supervised_flag)
                    tf.summary.scalar("labeled_pixels", sup_no)

                    if epoch >= SAVE_WTS_AFTR_EPOCH:
                        var = np.abs(cur_pred - ensemble_prediction)
                        mean_along_zone = np.mean(var, axis=(0, 1, 2, 3))
                        # next_weight = np.clip(next_weight, a_max=5, a_min=1)
                        unsupervised_weight = np.where(var <= mean_along_zone, np.ones_like(ensemble_prediction),
                                                       np.zeros_like(ensemble_prediction))  # consider hard labels
                        del ensemble_prediction
                        unsupervised_weight[:, :, :, :, 3] = unsupervised_weight[:, :, :, :, 3] * 4
                        # unsupervised_weight = np.where(var >= mean_along_zone,1., 1.)
                        # del mean_along_zone, var
                        save_array(self.weight_path, unsupervised_weight, start, end)
                        del unsupervised_weight
                    # del unsupervised_weight, ensemble_prediction
                    # del ensemble_prediction

                if 'cur_pred' in locals(): del cur_pred

                # shuffle and init datagen again
                np.random.shuffle(self.train_idx_list)
                DataGenerator(self.imgs_path,
                              self.gt_path,
                              self.ensemble_path,
                              self.weight_path,
                              self.supervised_flag,
                              self.train_idx_list,
                              **params)

    # callbacks
    print('-' * 30)
    print('Creating callbacks...')
    print('-' * 30)
    # csv_logger = CSVLogger('validation.csv', append=True, separator=';')
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

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=True, histogram_freq=1,
                              batch_size=2, write_images=False)

    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(0, num_train_data)]

    tcb = TemporalCallback(TRAIN_IMGS_PATH, TRAIN_GT_PATH, ENS_GT_PATH, WEIGHT_PATH, supervised_flag,
                           VAR_THRESHOLD, train_id_list)
    lcb = wm.LossCallback()
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    training_generator = DataGenerator(TRAIN_IMGS_PATH,
                                       TRAIN_GT_PATH,
                                       ENS_GT_PATH,
                                       WEIGHT_PATH,
                                       supervised_flag,
                                       train_id_list)

    del supervised_flag
    steps = num_train_data / batch_size
    # steps=2

    val_supervised_flag = np.ones((num_val_data, 32, 168, 168, 1), dtype='int8')
    val_unsupervised_weight = np.zeros((num_val_data, 32, 168, 168, 5), dtype='float32')
    val_x_arr = get_complete_array(VAL_IMGS_PATH)
    val_y_arr = get_complete_array(VAL_GT_PATH, data_type='int8')

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
    wm = weighted_model()
    model = wm.build_model(num_class=NUM_CLASS, use_dice_cl=False, learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=MODEL_NAME)
    print('load_weights')
    # model.load_weights()
    print('predict')
    out = model.predict(x_val, batch_size=1, verbose=1)

    print(model.evaluate(x_val, y_val, batch_size=1, verbose=1))

    np.save('predicted_sl2' + '.npy', out)


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = 2
    gpu_id = '0, 1'
    # gpu_id = '0'
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

    # train(None, None)
    # train(gpu, nb_gpus)
    # val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_imgs.npy')
    val_y = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')
    predict(val_x, val_y)