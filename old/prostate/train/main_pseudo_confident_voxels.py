import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, TensorBoard

from old.preprocess_images import get_complete_array, get_array, save_array
from old.prostate import weighted_model
from old.prostate.generator import DataGenerator
from old.utils.AugmentationGenerator import *
from old.utils.ops import ramp_down_weight
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

# 294 Training 58 have gt
learning_rate = 2.5e-5
TB_LOG_DIR = '/home/suhita/zonals/temporal/tb/variance_mcdropout/pseudo' + str(learning_rate) + '_confident_v/'
MODEL_NAME = '/home/suhita/zonals/temporal/pseudo_conf_voxel.h5'

TRAIN_IMGS_PATH = '/home/suhita/zonals/data/training/imgs/'
TRAIN_GT_PATH = '/home/suhita/zonals/data/training/gt/'

VAL_IMGS_PATH = '/home/suhita/zonals/data/test_anneke/imgs/'
VAL_GT_PATH = '/home/suhita/zonals/data/test_anneke/gt/'

TRAINED_MODEL_PATH = '/home/suhita/zonals/data/training_scripts.h5'
# TRAINED_MODEL_PATH = '/home/suhita/zonals/temporal/temporal_sl2.h5'

FLAG_PATH = '/home/suhita/zonals/temporal/sadv2/flag/'

NUM_CLASS = 5
num_epoch = 351
batch_size = 1
IMGS_PER_ENS_BATCH = 100

# hyper-params
# UPDATE_WTS_AFTER_EPOCH = 1
ENSEMBLE_NO = 1
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 100
# weight_max = 40
weight_max = 30

alpha = 0.6
VAR_THRESHOLD = 0.5

AFS = 3
def train(gpu_id, nb_gpus, trained_model=TRAINED_MODEL_PATH):
    num_labeled_train = 58
    num_train_data = len(os.listdir(TRAIN_IMGS_PATH))
    num_un_labeled_train = num_train_data - num_labeled_train
    num_val_data = len(os.listdir(VAL_IMGS_PATH))



    # prepare weights and arrays for updates
    #gen_weight = ramp_up_weight(ramp_up_period, weight_max * (num_labeled_train / num_train_data))
    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model

    wm = weighted_model()
    model = wm.build_model(num_class=NUM_CLASS, use_dice_cl=False, learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=trained_model)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling training_scripts...')
    print('-' * 30)

    # training_scripts.metrics_tensors += training_scripts.outputs
    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, imgs_path, gt_path, supervised_flag_path, train_idx_list):

            self.val_afs_dice_coef = 0.
            self.val_bg_dice_coef = 0.
            self.val_cz_dice_coef = 0.
            self.val_pz_dice_coef = 0.
            self.val_us_dice_coef = 0.

            self.imgs_path = imgs_path
            self.gt_path = gt_path
            self.supervised_flag_path = supervised_flag_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            flag = np.ones((32, 168, 168, 1)).astype('int8')

            for patient in np.arange(num_train_data):
                if patient < num_labeled_train:
                    np.save(self.supervised_flag_path + str(patient) + '.npy', flag)
                else:
                    np.save(self.supervised_flag_path + str(patient) + '.npy',
                            np.zeros((32, 168, 168, 1)).astype('int8'))

        def on_batch_begin(self, batch, logs=None):
            pass


        def on_epoch_begin(self, epoch, logs=None):
            # tf.summary.scalar("labeled_pixels", np.count_nonzero(self.supervised_flag))
            if epoch > ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))

        def dice_coef(self, y_true, y_pred, smooth=1.):

            y_true_f = tf.to_float(K.flatten(y_true))
            y_pred_f = tf.to_float(K.flatten(y_pred))
            intersection = K.sum(y_true_f * y_pred_f)

            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def on_epoch_end(self, epoch, logs={}):

            if epoch <= 100:
                THRESHOLD = 0.99
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
                    supervised_flag = get_array(self.supervised_flag_path, start, end, dtype='int8')

                    inp = [imgs, supervised_flag]
                    del imgs

                    cur_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))

                    model_out = model.predict(inp, batch_size=2, verbose=1)  # 1
                    # model_out = np.add(model_out, training_scripts.predict(inp, batch_size=2, verbose=1))  # 2
                    del inp

                    cur_pred[:, :, :, :, 0] = model_out[0]
                    cur_pred[:, :, :, :, 1] = model_out[1]
                    cur_pred[:, :, :, :, 2] = model_out[2]
                    cur_pred[:, :, :, :, 3] = model_out[3]
                    cur_pred[:, :, :, :, 4] = model_out[4]
                    del model_out

                    if b_no == 0:

                        y_true = get_array(TRAIN_GT_PATH, 0, 58, dtype='int8')
                        logs['pz_dice_coef'] = K.eval(self.dice_coef(y_true[:, :, :, :, 0], cur_pred[0:58, :, :, :, 0]))
                        logs['cz_dice_coef'] = K.eval(self.dice_coef(y_true[:, :, :, :, 1], cur_pred[0:58, :, :, :, 1]))
                        logs['us_dice_coef'] = K.eval(self.dice_coef(y_true[:, :, :, :, 2], cur_pred[0:58, :, :, :, 2]))
                        logs['afs_dice_coef'] = K.eval(self.dice_coef(y_true[:, :, :, :, 3], cur_pred[0:58, :, :, :, 3]))
                        logs['bg_dice_coef'] = K.eval(self.dice_coef(y_true[:, :, :, :, 4], cur_pred[0:58, :, :, :, 4]))


                        flag = supervised_flag[num_labeled_train:actual_batch_size]
                        flag = np.where(
                            np.reshape(np.max(cur_pred[num_labeled_train:actual_batch_size], axis=-1),
                                       flag.shape) >= THRESHOLD, np.ones_like(flag), np.zeros_like(flag))
                        save_array(self.supervised_flag_path, flag, num_labeled_train, actual_batch_size)

                    else:
                        flag = np.where(
                            np.reshape(np.max(cur_pred, axis=-1),
                                       supervised_flag.shape) >= THRESHOLD, np.ones_like(supervised_flag),
                            np.zeros_like(supervised_flag))
                        save_array(self.supervised_flag_path, flag, start, end)
                    del flag
                    del supervised_flag

                if 'cur_pred' in locals(): del cur_pred

                # shuffle and init datagen again
                np.random.shuffle(self.train_idx_list)
                DataGenerator(self.imgs_path,
                              self.gt_path,
                              self.supervised_flag_path,
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

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=True, histogram_freq=0,
                              batch_size=2, write_images=False)

    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(0, num_train_data)]

    tcb = TemporalCallback(TRAIN_IMGS_PATH, TRAIN_GT_PATH, FLAG_PATH, train_id_list)
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, LRDecay]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting training_scripts...')
    print('-' * 30)
    training_generator = DataGenerator(TRAIN_IMGS_PATH,
                                       TRAIN_GT_PATH,
                                       FLAG_PATH,
                                       train_id_list)


    steps = num_train_data / batch_size
    #steps =2

    val_supervised_flag = np.ones((num_val_data, 32, 168, 168, 1), dtype='int8')
    val_x_arr = get_complete_array(VAL_IMGS_PATH)
    val_y_arr = get_complete_array(VAL_GT_PATH, dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr, val_supervised_flag]
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
    val_supervised_flag = np.ones((val_x_arr.shape[0], 32, 168, 168, 1), dtype='int8')
    val_unsupervised_weight = np.ones((val_x_arr.shape[0], 32, 168, 168, 5), dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr, val_y_arr, val_supervised_flag, val_unsupervised_weight]
    wm = weighted_model()
    model = wm.build_model(num_class=NUM_CLASS, use_dice_cl=True, learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=MODEL_NAME)
    print('load_weights')
    # training_scripts.load_weights()
    print('predict')
    out = model.predict(x_val, batch_size=1, verbose=1)

    print(model.evaluate(x_val, y_val, batch_size=1, verbose=1))

    np.save('predicted_sl2' + '.npy', out)


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = batch_size
    gpu_id = '2'
    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    '''
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)
        '''

    train(None, None, trained_model=TRAINED_MODEL_PATH)
    #train(gpu, nb_gpus)
    # val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_imgs.npy')
    val_y = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy').astype('int8')
    predict(val_x, val_y)
