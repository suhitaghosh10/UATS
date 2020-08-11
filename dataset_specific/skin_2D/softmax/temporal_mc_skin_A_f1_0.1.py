import getpass

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

USER_NAME = getpass.getuser()
import sys

if USER_NAME == 'anneke':
    module_root = '../../'
    sys.path.append(module_root)

from dataset_specific.skin_2D import DataGenerator as train_gen
from dataset_specific.skin_2D import weighted_model
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel
from old.utils.preprocess_images import get_array, save_array
from old.utils.AugmentationGenerator import *
from shutil import copyfile
from dataset_specific.kits import makedir
import shutil

# learning_rate = 1e-7
AUGMENTATION_NO = 5
TEMP = 1
augmentation = True
FOLD_NUM = 1
PERCENTAGE_OF_PIXELS = 50
if USER_NAME == 'suhita':
    ENS_GT_PATH = '/data/suhita/temporal/skin/output/sm_mc/sadv222/'
elif USER_NAME == 'anneke':
    DATA_ROOT = '/home/anneke/data/skin'
    ENS_GT_PATH = DATA_ROOT + '/ensembleGT'
num_epoch = 1000
batch_size = 8
DIM = [192, 256]
N_CHANNELS = 3
alpha = 0.6

# hyper-params
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 50


def train(gpu_id, nb_gpus, perc, batch_nos, learning_rate=None, wts=None):
    if USER_NAME == 'suhita':
        DATA_PATH = '/cache/suhita/data/skin/softmax/fold_' + str(FOLD_NUM) + '_P' + str(perc) + '/'
        FOLD_DIR = '/cache/suhita/skin/Folds'
        TRAIN_NUM = len(np.load(FOLD_DIR + '/train_fold' + str(FOLD_NUM) + '.npy'))
        NAME = 'sm_skin_entropy_F' + str(FOLD_NUM) + '_Perct_Labelled_' + str(perc)

        TB_LOG_DIR = '/data/suhita/temporal/tb/' + NAME + '_' + str(learning_rate) + '/'
        MODEL_NAME = '/data/suhita/skin/models/' + NAME + '.h5'

        CSV_NAME = '/data/suhita/temporal/CSV/' + NAME + '.csv'

        TRAINED_MODEL_PATH = '/data/suhita/skin/models/softmax_supervised_sfs32_F_' + str(FOLD_NUM) + \
                             '_1000_5e-05_Perc_' + str(perc) + '_augm.h5'



    elif USER_NAME == 'anneke':
        DATA_PATH = DATA_ROOT + '/arrays/fold_' + str(FOLD_NUM) + '_P' + str(perc) + '/'
        FOLD_DIR = '/home/anneke/projects/zones_UATS/Temporal_Thesis/skin_2D/Folds'
        TRAIN_NUM = len(np.load(FOLD_DIR + '/train_fold' + str(FOLD_NUM) + '.npy'))
        NAME = 'sm_skin_entropy_F' + str(FOLD_NUM) + '_Perct_Labelled_' + str(perc)

        TB_LOG_DIR = DATA_ROOT + '/temporal_models/tb/' + NAME + '_' + str(learning_rate) + '/'
        MODEL_NAME = DATA_ROOT + '/temporal_models/' + NAME + '.h5' if wts is None else wts

        CSV_NAME = DATA_ROOT + '/temporal_models/' + NAME + '.csv'

        TRAINED_MODEL_PATH = DATA_ROOT + '/supervised_models/softmax_supervised_sfs32_F_' + str(
            FOLD_NUM) + '_1000_5e-05_Perc_' + str(perc) + '_augm.h5' if wts is None else wts

    num_labeled_train = int(perc * TRAIN_NUM)
    num_train_data = len(os.listdir(DATA_PATH + '/imgs/'))
    num_un_labeled_train = num_train_data - num_labeled_train
    IMGS_PER_ENS_BATCH = num_un_labeled_train // batch_nos
    # num_val_data = len(os.listdir(VAL_IMGS_PATH))

    # gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model, model_MC = wm.build_model(img_shape=(DIM[0], DIM[1], N_CHANNELS), learning_rate=learning_rate, gpu_id=gpu_id,
                                     nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, data_path, ensemble_path, train_idx_list):

            self.val_bg_dice_coef = 0.
            self.val_skin_dice_coef = 0.

            self.data_path = data_path
            self.ensemble_path = ensemble_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            # self.confident_pixels_no = (PERCENTAGE_OF_PIXELS * DIM[0] * DIM[1] * DIM[2] * IMGS_PER_ENS_BATCH *2 ) // 100
            flag = np.ones((*DIM, 1)).astype('float16')
            if os.path.exists(self.ensemble_path):
                raise Exception('the path exists!', self.ensemble_path)
            else:
                makedir(self.ensemble_path)
                makedir(os.path.join(self.ensemble_path, 'ens_gt'))
                makedir(os.path.join(self.ensemble_path, 'flag'))
            for patient in np.arange(num_train_data):

                copyfile(os.path.join(DATA_PATH, 'GT', str(patient) + '.npy'),
                         os.path.join(self.ensemble_path, 'ens_gt', str(patient) + '.npy'))

                if patient < num_labeled_train:
                    np.save(os.path.join(self.ensemble_path, 'flag', str(patient) + '.npy'), flag)
                else:
                    np.save(os.path.join(self.ensemble_path, 'flag', str(patient) + '.npy'),
                            np.zeros((*DIM, 1)).astype('float32'))

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
            '''
            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))
            # print(K.eval(model.layers[43].trainable_weights[0]))
'''
            pass

        def on_epoch_end(self, epoch, logs={}):
            # print(time() - self.starttime)
            # model_temp = model

            bg_save, self.val_bg_dice_coef = self.shall_save(logs['val_bg_dice_coef'], self.val_bg_dice_coef)
            lesion_save, self.val_skin_dice_coef = self.shall_save(logs['val_skin_dice_coef'], self.val_skin_dice_coef)
            model_MC.set_weights(model.get_weights())

            if epoch > 0:

                patients_per_batch = IMGS_PER_ENS_BATCH
                num_batches = num_un_labeled_train // patients_per_batch
                remainder = num_un_labeled_train % patients_per_batch
                remainder_pixels = remainder * DIM[0] * DIM[1]
                confident_pixels_no_per_batch = (PERCENTAGE_OF_PIXELS * patients_per_batch * DIM[0] * DIM[1]) // 100
                if remainder_pixels < confident_pixels_no_per_batch:
                    patients_per_last_batch = patients_per_batch + remainder
                else:
                    patients_per_last_batch = remainder
                    num_batches = num_batches + 1

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if (
                            b_no < num_batches - 1) else patients_per_last_batch
                    confident_pixels_no = (PERCENTAGE_OF_PIXELS * DIM[0] * DIM[1] * actual_batch_size) // 100
                    start = (b_no * patients_per_batch) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array(self.data_path + '/imgs/', start, end)
                    ensemble_prediction = get_array(self.ensemble_path + '/ens_gt/', start, end, dtype='float32')
                    supervised_flag = get_array(self.ensemble_path + '/flag/', start, end, dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, DIM[0], DIM[1], 2))
                    model_out = model.predict(inp, batch_size=batch_size, verbose=1)  # 1

                    # model_out = np.add(model_out, train.predict(inp, batch_size=2, verbose=1))  # 2
                    # del inp

                    cur_pred[:, :, :, 0] = model_out[0] if bg_save else ensemble_prediction[:, :, :, 0]
                    cur_pred[:, :, :, 1] = model_out[1] if lesion_save else ensemble_prediction[:, :, :, 1]
                    mc_pred = np.zeros((actual_batch_size, DIM[0], DIM[1], 2))

                    del model_out

                    # Z = αZ + (1 - α)z
                    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred
                    save_array(self.ensemble_path, ensemble_prediction, start, end)
                    del ensemble_prediction

                    # mc dropout chnages
                    T = 10
                    for i in np.arange(T):
                        model_out = model_MC.predict(inp, batch_size=batch_size, verbose=1)

                        mc_pred[:, :, :, 0] = np.add(model_out[0], mc_pred[:, :, :, 0])
                        mc_pred[:, :, :, 1] = np.add(model_out[1], mc_pred[:, :, :, 1])

                    # avg_pred = mc_pred / T#
                    entropy = None
                    for z in np.arange(2):
                        if z == 0:
                            entropy = (mc_pred[:, :, :, z] / T) * np.log((mc_pred[:, :, :, z] / T) + 1e-5)
                        else:
                            entropy = entropy + (mc_pred[:, :, :, z] / T) * np.log(
                                (mc_pred[:, :, :, z] / T) + 1e-5)
                    entropy = -entropy
                    del mc_pred, inp, model_out

                    argmax_pred_ravel = np.ravel(np.argmin(cur_pred, axis=-1))
                    max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))

                    indices = None
                    del cur_pred
                    for zone in np.arange(2):
                        entropy_zone = np.ravel(entropy[:, :, :])
                        final_max_ravel = np.where(argmax_pred_ravel == zone, np.zeros_like(entropy_zone),
                                                   entropy_zone)
                        zone_indices = np.argpartition(final_max_ravel, -confident_pixels_no)[
                                       -confident_pixels_no:]
                        if zone == 0:
                            indices = zone_indices
                        else:
                            indices = np.unique(np.concatenate((zone_indices, indices)))

                    mask = np.ones(entropy_zone.shape, dtype=bool)
                    mask[indices] = False

                    entropy_zone[mask] = 0
                    entropy_zone = np.where(entropy_zone > 0, np.ones_like(entropy_zone) * 2,
                                            np.zeros_like(entropy_zone))
                    flag = np.reshape(max_pred_ravel, (actual_batch_size, DIM[0], DIM[1], 1))
                    del entropy_zone, indices

                    save_array(self.ensemble_path + '/flag/', flag, start, end)
                    del flag

                if 'cur_pred' in locals(): del cur_pred

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

    print(train_id_list[0:10])

    np.random.shuffle(train_id_list)
    tcb = TemporalCallback(DATA_PATH, ENS_GT_PATH, train_id_list)
    lcb = wm.LossCallback()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0005)
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger, es]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    # params = {'dim': (32, 168, 168),'batch_size': batch_size}

    print('-' * 30)
    print('Fitting train...')
    print('-' * 30)
    training_generator = train_gen(DATA_PATH,
                                   ENS_GT_PATH,
                                   train_id_list,
                                   batch_size=batch_size,
                                   augmentation=True)

    # steps = num_train_data / batch_size
    if augmentation == False:
        augm_no = 1
    else:
        augm_no = AUGMENTATION_NO
    steps = (num_train_data * augm_no) / batch_size
    # steps = 2

    val_fold = np.load(FOLD_DIR + '/val_fold' + str(FOLD_NUM) + '.npy')
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data, DIM[0], DIM[1], 1), dtype='int8')
    val_img_arr = np.zeros((num_val_data, DIM[0], DIM[1], 3), dtype=float)
    val_GT_arr = np.zeros((num_val_data, DIM[0], DIM[1], 2), dtype=float)
    if USER_NAME == 'suhita':
        VAL_DATA = '/cache/suhita/skin/preprocessed/labelled/train'
    elif USER_NAME == 'anneke':
        VAL_DATA = '/home/anneke/data/skin/preprocessed/labelled/train'
    for i in np.arange(num_val_data):
        val_img_arr[i] = np.load(os.path.join(VAL_DATA, 'imgs', val_fold[i]))
        val_GT_arr[i, :, :, 1] = np.load(
            os.path.join(VAL_DATA, 'GT', val_fold[i]).replace('.npy', '_segmentation.npy'))[:, :, 0]
        val_GT_arr[i, :, :, 1] = val_GT_arr[i, :, :, 1] / 255
        val_GT_arr[i, :, :, 0] = np.where(val_GT_arr[i, :, :, 1] == 0, np.ones_like(val_GT_arr[i, :, :, 1]),
                                          np.zeros_like(val_GT_arr[i, :, :, 1]))
    val_img_arr = val_img_arr / 255

    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr[:, :, :, 0], val_GT_arr[:, :, :, 1]]
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )
    if 'model' in locals(): del model
    if 'model_MC' in locals(): del model_MC


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = batch_size
    gpu_id = '0'

    # gpu_id = '0'
    gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.compat.v1.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    try:
        # train(None, None, perc=0.05, batch_nos=5, learning_rate=1e-6,
        #      wts=DATA_ROOT+'/temporal_models/sm_skin_entropy_F' + str(FOLD_NUM) + '_Perct_Labelled_0.05.h5')
        # shutil.rmtree(ENS_GT_PATH)
        train(None, None, perc=0.1, batch_nos=5, learning_rate=1e-6)
    # shutil.rmtree(ENS_GT_PATH)
    # train(None, None, perc=0.25, batch_nos=5, learning_rate=1e-6)
    # shutil.rmtree(ENS_GT_PATH)
    # train(None, None, perc=0.5, batch_nos=4, learning_rate=1e-7)
    # shutil.rmtree(ENS_GT_PATH)
    # train(None, None, perc=1.0, batch_nos=3, learning_rate=1e-7)

    finally:

        if os.path.exists(ENS_GT_PATH):
            shutil.rmtree(ENS_GT_PATH)
        print('clean up done!')
