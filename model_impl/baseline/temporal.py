import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from generator.temporal_A import DataGenerator
from lib.segmentation.model.temporal_not_scaled import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_complete_array, get_array, save_array
from zonal_utils.AugmentationGenerator import *

# 294 Training 58 have gt
learning_rate = 5e-5

TRAIN_NUM = 58
FOLD_NUM = 2
CSV_NAME = '/data/suhita/temporal/CSV/Temporal_Supervised_F' + str(FOLD_NUM) + '.csv'
NAME = 'Temporal_Supervised_F' + str(FOLD_NUM)
TB_LOG_DIR = '/data/suhita/temporal/tb/variance_mcdropout/' + NAME + '_' + str(learning_rate) + '/'
MODEL_NAME = '/data/suhita/temporal/' + NAME + '.h5'

TRAIN_IMGS_PATH = '/cache/suhita/data/fold2/train/imgs/'
TRAIN_GT_PATH = '/cache/suhita/data/fold2/train/gt/'

VAL_IMGS_PATH = '/cache/suhita/data/fold2/val/imgs/'
VAL_GT_PATH = '/cache/suhita/data/fold2/val/gt/'

ENS_GT_PATH = '/data/suhita/temporal/sadv3/ens_gt/'
FLAG_PATH = '/data/suhita/temporal/sadv3/flag/'

TRAINED_MODEL_PATH = None

NUM_CLASS = 5
num_epoch = 351
batch_size = 2


def train(gpu_id, nb_gpus):
    num_labeled_train = 58
    num_train_data = len(os.listdir(TRAIN_IMGS_PATH))
    num_val_data = len(os.listdir(VAL_IMGS_PATH))

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model = wm.build_model(num_class=NUM_CLASS, learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH)
    print("Images Size:", num_train_data)

    print('-' * 30)
    print('Creating and compiling model_impl...')
    print('-' * 30)

    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, imgs_path, gt_path, ensemble_path, supervised_flag_path, train_idx_list):

            self.val_afs_dice_coef = 0.
            self.val_bg_dice_coef = 0.
            self.val_cz_dice_coef = 0.
            self.val_pz_dice_coef = 0.
            self.val_us_dice_coef = 0.
            self.imgs_path = imgs_path
            self.gt_path = gt_path
            self.ensemble_path = ensemble_path
            self.supervised_flag_path = supervised_flag_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg

            unsupervised_target = get_complete_array(TRAIN_GT_PATH, dtype='float32')
            flag = np.ones((32, 168, 168)).astype('int8')

            for patient in np.arange(num_train_data):
                np.save(self.ensemble_path + str(patient) + '.npy', unsupervised_target[patient])
                np.save(self.supervised_flag_path + str(patient) + '.npy', flag)
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
            pass

        def on_epoch_end(self, epoch, logs={}, alpha=0.6):
            pz_save, self.val_pz_dice_coef = self.shall_save(logs['val_pz_dice_coef'], self.val_pz_dice_coef)
            cz_save, self.val_cz_dice_coef = self.shall_save(logs['val_cz_dice_coef'], self.val_cz_dice_coef)
            us_save, self.val_us_dice_coef = self.shall_save(logs['val_us_dice_coef'], self.val_us_dice_coef)
            afs_save, self.val_afs_dice_coef = self.shall_save(logs['val_afs_dice_coef'], self.val_afs_dice_coef)
            bg_save, self.val_bg_dice_coef = self.shall_save(logs['val_bg_dice_coef'], self.val_bg_dice_coef)

            if epoch >= 0:

                actual_batch_size = num_labeled_train
                start = 0
                end = actual_batch_size
                imgs = get_array(self.imgs_path, start, end)
                ensemble_prediction = get_array(self.ensemble_path, start, end, dtype='float32')
                supervised_flag = get_array(self.supervised_flag_path, start, end, dtype='float16')

                inp = [imgs, ensemble_prediction, supervised_flag]
                del imgs, supervised_flag

                cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))

                model_out = model.predict(inp, batch_size=2, verbose=1)
                del inp

                cur_pred[:, :, :, :, 0] = model_out[0] if pz_save else ensemble_prediction[:, :, :, :, 0]
                cur_pred[:, :, :, :, 1] = model_out[1] if cz_save else ensemble_prediction[:, :, :, :, 1]
                cur_pred[:, :, :, :, 2] = model_out[2] if us_save else ensemble_prediction[:, :, :, :, 2]
                cur_pred[:, :, :, :, 3] = model_out[3] if afs_save else ensemble_prediction[:, :, :, :, 3]
                cur_pred[:, :, :, :, 4] = model_out[4] if bg_save else ensemble_prediction[:, :, :, :, 4]

                del model_out

                # Z = αZ + (1 - α)z
                ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred
                save_array(self.ensemble_path, ensemble_prediction, start, end)
                del ensemble_prediction

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

    # only consider data having gt (no pseudolabels)
    train_id_list = [str(i) for i in np.arange(0, num_labeled_train)]

    tcb = TemporalCallback(TRAIN_IMGS_PATH, TRAIN_GT_PATH, ENS_GT_PATH, FLAG_PATH, train_id_list)
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
    print('Fitting model_impl...')
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
    # model_impl.save('temporal_max_ramp_final.h5')


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
    model = wm.build_model(num_class=NUM_CLASS, use_dice_cl=True, learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=MODEL_NAME, temp=1)
    print('load_weights')
    # model_impl.load_weights()
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
    # val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_imgs.npy')
    val_y = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')
    predict(val_x, val_y)
