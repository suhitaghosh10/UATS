import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint, TensorBoard

from old.preprocess_images import get_complete_array, get_array, save_array
from old.utils.AugmentationGenerator import *
from prostate.generator.uats_A import DataGenerator
from prostate.model import weighted_model
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

# CHANGE BEFORE YOU RUN
learning_rate = 5e-5
FOLD_NUM = 1
TB_LOG_DIR = '/data/suhita/temporal/tb/variance_mcdropout/pseudo_5per_A_F' + str(FOLD_NUM) + '_' + str(
    learning_rate) + '/'
MODEL_NAME = '/data/suhita/temporal/pseudo_5per_A_F' + str(FOLD_NUM)

TRAIN_IMGS_PATH = '/cache/suhita/data/fold/' + str(FOLD_NUM) + '/train/imgs/'
TRAIN_GT_PATH = '/cache/suhita/data/fold' + str(FOLD_NUM) + '/train/gt/'
VAL_IMGS_PATH = '/cache/suhita/data/fold/' + str(FOLD_NUM) + '/val/imgs/'
VAL_GT_PATH = '/cache/suhita/data/fold/' + str(FOLD_NUM) + '/val/imgs/'
TRAINED_MODEL_PATH = '/data/suhita/temporal/supervised_F' + str(FOLD_NUM) + '.h5'
ENS_GT_PATH = '/data/suhita/temporal/sadv2/ens_gt/'
FLAG_PATH = '/data/suhita/temporal/sadv2/flag/'
CSV = '/data/suhita/temporal/CSV/pseudo_5per_A_F' + str(FOLD_NUM) + '.csv'
PERCENTAGE_OF_PIXELS = 5

NUM_CLASS = 5
num_labeled_train = 58
num_epoch = 1000
batch_size = 2
IMGS_PER_ENS_BATCH = 59


def train(gpu_id, nb_gpus):
    num_train_data = len(os.listdir(TRAIN_IMGS_PATH))
    num_un_labeled_train = num_train_data - num_labeled_train
    num_val_data = len(os.listdir(VAL_IMGS_PATH))

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model = wm.build_model(learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH
                           )
    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling training_scripts...')
    print('-' * 30)

    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, imgs_path, gt_path, ensemble_path, supervised_flag_path, train_idx_list):

            self.imgs_path = imgs_path
            self.gt_path = gt_path
            self.ensemble_path = ensemble_path
            self.supervised_flag_path = supervised_flag_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            self.confident_pixels_no = (PERCENTAGE_OF_PIXELS * 168 * 168 * 32 * num_un_labeled_train) // 100

            unsupervised_target = get_complete_array(TRAIN_GT_PATH, dtype='float32')
            flag = np.ones((32, 168, 168)).astype('int8')

            for patient in np.arange(num_train_data):
                np.save(self.ensemble_path + str(patient) + '.npy', unsupervised_target[patient])
                if patient < num_labeled_train:
                    np.save(self.supervised_flag_path + str(patient) + '.npy', flag)
                else:
                    np.save(self.supervised_flag_path + str(patient) + '.npy',
                            np.zeros((32, 168, 168)).astype('int8'))
            del unsupervised_target

        def shall_save(self, cur_val, prev_val):
            flag_save = False
            val_save = prev_val

            if cur_val > prev_val:
                flag_save = True
                val_save = cur_val

            return flag_save, val_save

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs={}):

            if epoch >= 0:

                patients_per_batch = IMGS_PER_ENS_BATCH
                num_batches = num_un_labeled_train // patients_per_batch
                remainder = num_un_labeled_train % patients_per_batch
                num_batches = num_batches if remainder is 0 else num_batches + 1

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if (
                                b_no <= num_batches - 1 and remainder == 0) else remainder
                    start = (b_no * patients_per_batch) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array(self.imgs_path, start, end)
                    ensemble_prediction = get_array(self.ensemble_path, start, end, dtype='float32')
                    supervised_flag = get_array(self.supervised_flag_path, start, end, dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))

                    model_out = model.predict(inp, batch_size=2, verbose=1)
                    del inp

                    cur_pred[:, :, :, :, 0] = model_out[0]
                    cur_pred[:, :, :, :, 1] = model_out[1]
                    cur_pred[:, :, :, :, 2] = model_out[2]
                    cur_pred[:, :, :, :, 3] = model_out[3]
                    cur_pred[:, :, :, :, 4] = model_out[4]
                    del model_out

                    argmax_pred_ravel = np.ravel(np.argmax(cur_pred, axis=-1))
                    max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))
                    indices = None
                    del cur_pred
                    for zone in np.arange(4):
                        final_max_ravel = np.where(argmax_pred_ravel == zone, np.zeros_like(max_pred_ravel),
                                                   max_pred_ravel)
                        zone_indices = np.argpartition(final_max_ravel, -self.confident_pixels_no)[
                                       -self.confident_pixels_no:]
                        if zone == 0:
                            indices = zone_indices
                        else:
                            indices = np.unique(np.concatenate((zone_indices, indices)))

                    mask = np.ones(max_pred_ravel.shape, dtype=bool)
                    mask[indices] = False

                    max_pred_ravel[mask] = 0
                    flag = np.reshape(max_pred_ravel, (IMGS_PER_ENS_BATCH, 32, 168, 168))
                    del max_pred_ravel, indices

                    save_array(self.supervised_flag_path, flag, start, end)

                    del flag

                if 'cur_pred' in locals(): del cur_pred

                # shuffle and init datagen again

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

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=True, histogram_freq=0,
                              batch_size=2, write_images=False)
    csv_logger = CSVLogger(CSV, append=True, separator=';')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(0, num_train_data)]

    tcb = TemporalCallback(TRAIN_IMGS_PATH, TRAIN_GT_PATH, ENS_GT_PATH, FLAG_PATH, train_id_list)
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)

    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, csv_logger, LRDecay, es]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting training_scripts...')
    print('-' * 30)
    training_generator = DataGenerator(TRAIN_IMGS_PATH,
                                       TRAIN_GT_PATH,
                                       ENS_GT_PATH,
                                       FLAG_PATH,
                                       train_id_list)

    steps = num_train_data / batch_size
    # steps =2

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

    history = model.fit_generator(generator=training_generator,
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
    model = wm.build_model(num_class=NUM_CLASS, learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=MODEL_NAME, temp=1)
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
