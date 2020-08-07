import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from kits.utils import makedir
from old.preprocess_images import get_complete_array, get_array, save_array
from old.utils.AugmentationGenerator import *
from old.utils.ops import ramp_down_weight
from prostate.generator.uats_A import DataGenerator
from prostate.model import weighted_model
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

# 294 Training 58 have gt
learning_rate = 5e-5

FOLD_NUM = 1
TB_LOG_DIR = '/data/suhita/temporal/tb/prostate/MC_Ent_F' + str(
    FOLD_NUM) + '_' + str(learning_rate) + '/'
MODEL_NAME = '/data/suhita/temporal/MC_Ent_F' + str(FOLD_NUM)

CSV_NAME = '/data/suhita/temporal/CSV/MC_Ent_F' + str(FOLD_NUM) + '.csv'

TRAIN_IMGS_PATH = '/cache/suhita/data/prostate/fold_1_P1.0/train/'
# TRAIN_UNLABELED_DATA_PRED_PATH = '/cache/suhita/data/training/ul_gt/'

VAL_IMGS_PATH = '/cache/suhita/data/prostate/fold_' + str(FOLD_NUM) + '_P1.0/val/imgs/'
VAL_GT_PATH = '/cache/suhita/data/prostate/fold_' + str(FOLD_NUM) + '_P1.0/val/gt/'

TRAINED_MODEL_PATH = '/data/suhita/prostate/supervised_F' + str(FOLD_NUM) + '_P1.0.h5'

ENS_GT_PATH = '/data/suhita/temporal/sad_mc_f1/'
makedir(ENS_GT_PATH)
makedir(ENS_GT_PATH + '/ens_gt/')
makedir(ENS_GT_PATH + '/flag/')
PERCENTAGE_OF_PIXELS = 25

NUM_CLASS = 5
num_epoch = 1000
batch_size = 2
IMGS_PER_ENS_BATCH = 59  # 236/4 = 59

# hyper-params
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 50
# weight_max = 40
weight_max = 30

alpha = 0.6


def train(gpu_id, nb_gpus):
    num_labeled_train = 58
    num_train_data = len(os.listdir(os.path.join(TRAIN_IMGS_PATH, 'imgs')))
    num_un_labeled_train = num_train_data - num_labeled_train
    num_val_data = len(os.listdir(VAL_IMGS_PATH))

    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model

    wm = weighted_model()
    merged_model, p_model_MC, model = wm.build_model(learning_rate=learning_rate, gpu_id=gpu_id,
                                                     nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling training_scripts...')
    print('-' * 30)

    merged_model.summary()

    class TemporalCallback(Callback):

        def __init__(self, imgs_path, ensemble_path, train_idx_list):

            self.val_afs_dice_coef = 0.
            self.val_bg_dice_coef = 0.
            self.val_cz_dice_coef = 0.
            self.val_pz_dice_coef = 0.
            self.val_us_dice_coef = 0.
            self.val_loss = 0.
            self.count = 58 * 168 * 168 * 32

            self.imgs_path = imgs_path
            self.ensemble_path = ensemble_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            self.confident_pixels_no = (PERCENTAGE_OF_PIXELS * 168 * 168 * 32 * num_un_labeled_train) // 100

            unsupervised_target = get_complete_array(TRAIN_IMGS_PATH + '/GT/', dtype='float32')
            flag = np.ones((32, 168, 168)).astype('float16')

            for patient in np.arange(num_train_data):
                np.save(self.ensemble_path + '/ens_gt/' + str(patient) + '.npy', unsupervised_target[patient])
                if patient < num_labeled_train:
                    np.save(self.ensemble_path + '/flag/' + str(patient) + '.npy', flag)
                else:
                    np.save(self.ensemble_path + '/flag/' + str(patient) + '.npy',
                            np.zeros((32, 168, 168)).astype('float16'))
            del unsupervised_target

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
            # tf.summary.scalar("labeled_pixels", np.count_nonzero(self.supervised_flag))
            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(merged_model.optimizer.lr, weight_down * learning_rate)
                K.set_value(merged_model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(merged_model.optimizer.lr), K.eval(merged_model.optimizer.beta_1))

        def on_epoch_end(self, epoch, logs={}):
            sup_count = self.count
            pz_save, self.val_pz_dice_coef = self.shall_save(logs['val_pz_dice_coef'], self.val_pz_dice_coef)
            cz_save, self.val_cz_dice_coef = self.shall_save(logs['val_cz_dice_coef'], self.val_cz_dice_coef)
            us_save, self.val_us_dice_coef = self.shall_save(logs['val_us_dice_coef'], self.val_us_dice_coef)
            afs_save, self.val_afs_dice_coef = self.shall_save(logs['val_afs_dice_coef'], self.val_afs_dice_coef)
            bg_save, self.val_bg_dice_coef = self.shall_save(logs['val_bg_dice_coef'], self.val_bg_dice_coef)

            if epoch > 0:

                num_batches = num_un_labeled_train // IMGS_PER_ENS_BATCH
                remainder = num_un_labeled_train % IMGS_PER_ENS_BATCH
                remainder_pixels = remainder * 168 * 168 * 32
                confident_pixels_no_per_batch = (PERCENTAGE_OF_PIXELS * IMGS_PER_ENS_BATCH * 168 * 168 * 32) // 100
                if remainder_pixels < confident_pixels_no_per_batch:
                    patients_per_last_batch = IMGS_PER_ENS_BATCH + remainder
                else:
                    patients_per_last_batch = remainder
                    num_batches = num_batches + 1
                update_flag = True if logs['val_loss'] < self.val_loss else False
                for b_no in np.arange(num_batches):
                    actual_batch_size = IMGS_PER_ENS_BATCH if (
                            b_no < num_batches - 1) else patients_per_last_batch
                    start = (b_no * IMGS_PER_ENS_BATCH) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array(self.imgs_path + '/imgs/', start, end)
                    ensemble_prediction = get_array(self.ensemble_path + '/ens_gt/', start, end, dtype='float32')
                    supervised_flag = get_array(self.ensemble_path + '/flag/', start, end, dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))
                    mc_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))

                    model_out = merged_model.predict(inp, batch_size=2, verbose=1)  # 1
                    # model_out = np.add(model_out, training_scripts.predict(inp, batch_size=2, verbose=1))  # 2
                    # del inp

                    cur_pred[:, :, :, :, 0] = model_out[0] if pz_save else ensemble_prediction[:, :, :, :, 0]
                    cur_pred[:, :, :, :, 1] = model_out[1] if cz_save else ensemble_prediction[:, :, :, :, 1]
                    cur_pred[:, :, :, :, 2] = model_out[2] if us_save else ensemble_prediction[:, :, :, :, 2]
                    cur_pred[:, :, :, :, 3] = model_out[3] if afs_save else ensemble_prediction[:, :, :, :, 3]
                    cur_pred[:, :, :, :, 4] = model_out[4] if bg_save else ensemble_prediction[:, :, :, :, 4]

                    mc_pred[:, :, :, :, 0] = model_out[5]
                    mc_pred[:, :, :, :, 1] = model_out[6]
                    mc_pred[:, :, :, :, 2] = model_out[7]
                    mc_pred[:, :, :, :, 3] = model_out[8]
                    mc_pred[:, :, :, :, 4] = model_out[9]

                    # del model_out

                    # Z = αZ + (1 - α)z
                    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred[:, :, :, :, 0:5]
                    # del cur_pred

                    save_array(self.ensemble_path + '/ens_gt/', ensemble_prediction, start, end)
                    del ensemble_prediction

                    if update_flag:
                        self.val_loss = logs['val_loss']
                        T = 20
                        for i in np.arange(T - 1):
                            model_out = merged_model.predict(inp, batch_size=2, verbose=1)

                            mc_pred[:, :, :, :, 0] = np.add(model_out[5], mc_pred[:, :, :, :, 0])
                            mc_pred[:, :, :, :, 1] = np.add(model_out[6], mc_pred[:, :, :, :, 1])
                            mc_pred[:, :, :, :, 2] = np.add(model_out[7], mc_pred[:, :, :, :, 2])
                            mc_pred[:, :, :, :, 3] = np.add(model_out[8], mc_pred[:, :, :, :, 3])
                            mc_pred[:, :, :, :, 4] = np.add(model_out[9], mc_pred[:, :, :, :, 4])

                        avg_pred = mc_pred / T
                        entropy = -(mc_pred / T) * np.log((mc_pred / T) + 1e-5)
                        del mc_pred, inp, model_out

                        argmax_pred_ravel = np.ravel(np.argmin(cur_pred, axis=-1))
                        max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))

                        indices = None
                        del cur_pred
                        for zone in np.arange(4):
                            entropy_zone = np.ravel(entropy[:, :, :, :, zone])
                            final_max_ravel = np.where(argmax_pred_ravel == zone, np.zeros_like(entropy_zone),
                                                       entropy_zone)
                            zone_indices = np.argpartition(final_max_ravel, -confident_pixels_no_per_batch)[
                                           -confident_pixels_no_per_batch:]
                            if zone == 0:
                                indices = zone_indices
                            else:
                                indices = np.unique(np.concatenate((zone_indices, indices)))

                        mask = np.ones(entropy_zone.shape, dtype=bool)
                        mask[indices] = False

                        entropy_zone[mask] = 0
                        entropy_zone = np.where(entropy_zone > 0, np.ones_like(entropy_zone) * 2,
                                                np.zeros_like(entropy_zone))
                        flag = np.reshape(entropy_zone, (actual_batch_size, 32, 168, 168))
                        del entropy_zone, indices

                        save_array(self.ensemble_path + '/flag/', flag, start, end)

                        sup_count = sup_count + np.count_nonzero(flag)
                        del flag

                if 'cur_pred' in locals(): del cur_pred

                # shuffle and init datagen again

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
                              batch_size=2, write_images=False)

    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(0, num_train_data)]

    tcb = TemporalCallback(TRAIN_IMGS_PATH, ENS_GT_PATH, train_id_list)
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)
    lcb = wm.LossCallback()
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb, LRDecay, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting training_scripts...')
    print('-' * 30)
    training_generator = DataGenerator(TRAIN_IMGS_PATH,
                                       ENS_GT_PATH,
                                       train_id_list)

    steps = num_train_data / batch_size
    #steps =2

    val_supervised_flag = np.ones((num_val_data, 32, 168, 168), dtype='int8')
    val_x_arr = get_complete_array(VAL_IMGS_PATH)
    val_y_arr = get_complete_array(VAL_GT_PATH, dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr, val_y_arr, val_supervised_flag]
    del val_supervised_flag, pz, cz, us, afs, bg, val_y_arr, val_x_arr
    history = merged_model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    # workers=4)
    # training_scripts.save('temporal_max_ramp_final.h5')


def predict(val_x_arr, val_y_arr):
    val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168), dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr, val_y_arr, val_supervised_flag]
    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=MODEL_NAME)
    print('load_weights')
    # training_scripts.load_weights()
    print('predict')
    out = model.predict(x_val, batch_size=1, verbose=1)
    print(model.metrics_names)
    print(model.evaluate(x_val, y_val, batch_size=1, verbose=1))

    np.save('predicted_sl2' + '.npy', out)


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = batch_size
    gpu_id = '3'
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

    train(None, None)
    # train(gpu, nb_gpus)
    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/cache/suhita/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = np.load('/cache/suhita/data/test_anneke/final_test_array_imgs.npy')
    val_y = np.load('/cache/suhita/data/test_anneke/final_test_array_GT.npy').astype('int8')
    predict(val_x, val_y)
