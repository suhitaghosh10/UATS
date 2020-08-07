from shutil import copyfile

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from kits import DataGenerator as train_gen
from kits import weighted_model
from kits.utils import makedir
from old.preprocess_images import get_array_kits, save_array_kits
from old.utils.AugmentationGenerator import *
from old.utils.ops import ramp_down_weight
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

# 294 Training 58 have gt
SEGM_RIGHT_NPY = 'segm_right.npy'
SEGM_LEFT_NPY = 'segm_left.npy'
learning_rate = 2.5e-5
AUGMENTATION_NO = 5
augmentation = True
DROPOUT_NUM = 20
# TEMP = 2.908655

FOLD_NUM = 1
PERCENTAGE_OF_PIXELS = 50
PERCENTAGE_OF_LABELLED = 0.1
DATA_PATH = '/cache/suhita/data/kits/fold_' + str(FOLD_NUM) + '_P' + str(PERCENTAGE_OF_LABELLED) + '/'
TRAIN_NUM = len(np.load('/data/suhita/temporal/kits/Folds/train_fold1.npy'))
NAME = 'kits_mc_F' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_Perct_Labelled_' + str(PERCENTAGE_OF_LABELLED)

TB_LOG_DIR = '/data/suhita/temporal/tb/kits/' + NAME + '_' + str(learning_rate) + '/'
MODEL_NAME = '/data/suhita/temporal/' + NAME + '.h5'

CSV_NAME = '/data/suhita/temporal/CSV/' + NAME + '.csv'

# TRAINED_MODEL_PATH = '/data/suhita/temporal/kits/models/' + str(FOLD_NUM) + '_supervised_Perc_' + str(PERCENTAGE_OF_LABELLED) + '.h5'
TRAINED_MODEL_PATH = MODEL_NAME
ENS_GT_PATH = '/data/suhita/temporal/kits/output/sadv3/'

NUM_CLASS = 1
num_epoch = 1000
batch_size = 2
IMGS_PER_ENS_BATCH = None
DIM = [152, 152, 56]

# hyper-params
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 50
# weight_max = 40
# weight_max = 30

alpha = 0.6


def train(gpu_id, nb_gpus):
    num_labeled_train = int(PERCENTAGE_OF_LABELLED * TRAIN_NUM)
    num_train_data = len(os.listdir(DATA_PATH))
    num_un_labeled_train = num_train_data - num_labeled_train
    IMGS_PER_ENS_BATCH = num_un_labeled_train // 3

    gen_lr_weight = ramp_down_weight(ramp_down_period)

    # prepare dataset
    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model, p_model_MC, normal_model = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate,
                                                     gpu_id=gpu_id,
                                                     nb_gpus=nb_gpus, trained_model=TRAINED_MODEL_PATH)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model.summary()

    class TemporalCallback(Callback):

        def __init__(self, data_path, ensemble_path, train_idx_list):

            self.val_dice_coef = 0.

            self.data_path = data_path
            self.ensemble_path = ensemble_path
            self.train_idx_list = train_idx_list  # list of indexes of training eg
            # self.confident_pixels_no = (PERCENTAGE_OF_PIXELS * DIM[0] * DIM[1] * DIM[2] * num_un_labeled_train * 2) // 100

            flag = np.ones((*DIM, 1)).astype('float16')
            if os.path.exists(self.ensemble_path):
                raise Exception('the path exists!', self.ensemble_path)
            else:
                makedir(self.ensemble_path)
            for patient in np.arange(num_train_data):

                makedir(os.path.join(self.ensemble_path, 'case_' + str(patient)))
                copyfile(os.path.join(DATA_PATH, 'case_' + str(patient), SEGM_LEFT_NPY),
                         os.path.join(self.ensemble_path, 'case_' + str(patient), SEGM_LEFT_NPY))
                copyfile(os.path.join(DATA_PATH, 'case_' + str(patient), SEGM_RIGHT_NPY),
                         os.path.join(self.ensemble_path, 'case_' + str(patient), SEGM_RIGHT_NPY))

                if patient < num_labeled_train:
                    np.save(self.ensemble_path + 'case_' + str(patient) + '/flag_left.npy', flag)
                    np.save(self.ensemble_path + 'case_' + str(patient) + '/flag_right.npy', flag)
                else:
                    np.save(self.ensemble_path + 'case_' + str(patient) + '/flag_left.npy',
                            np.zeros((*DIM, 1)).astype('float32'))
                    np.save(self.ensemble_path + 'case_' + str(patient) + '/flag_right.npy',
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
            if epoch > num_epoch - ramp_down_period:
                weight_down = next(gen_lr_weight)
                K.set_value(model.optimizer.lr, weight_down * learning_rate)
                K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)
                print('LR: alpha-', K.eval(model.optimizer.lr), K.eval(model.optimizer.beta_1))

        def on_epoch_end(self, epoch, logs={}):
            # print(time() - self.starttime)
            # model_temp = model

            save, self.val_dice_coef = self.shall_save(logs['val_dice_coef'], self.val_dice_coef)
            p_model_MC.set_weights(normal_model.get_weights())

            if epoch > 0:

                patients_per_batch = IMGS_PER_ENS_BATCH
                num_batches = num_un_labeled_train // patients_per_batch
                remainder = num_un_labeled_train % patients_per_batch
                remainder_pixels = remainder * DIM[0] * DIM[1] * DIM[2] * 2
                confident_pixels_no_per_batch = (PERCENTAGE_OF_PIXELS * 2 * patients_per_batch * DIM[0] * DIM[1] * DIM[
                    2]) // 100
                if remainder_pixels < confident_pixels_no_per_batch:
                    patients_per_last_batch = patients_per_batch + remainder
                else:
                    patients_per_last_batch = remainder
                    num_batches = num_batches + 1

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if (
                            b_no < num_batches - 1) else patients_per_last_batch
                    confident_pixels_no = (PERCENTAGE_OF_PIXELS * DIM[0] * DIM[1] * DIM[
                        2] * actual_batch_size * 2) // 100
                    start = (b_no * patients_per_batch) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array_kits(self.data_path, start, end, 'img')
                    ensemble_prediction = get_array_kits(self.ensemble_path, start, end, 'segm', dtype='float32')
                    supervised_flag = get_array_kits(self.ensemble_path, start, end, 'flag', dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, DIM[0], DIM[1], DIM[2], 1))
                    # cur_sigmoid_pred = np.zeros((actual_batch_size, 32, 168, 168, NUM_CLASS))
                    model_out = model.predict(inp, batch_size=2, verbose=1)  # 1

                    # model_out = np.add(model_out, training_scripts.predict(inp, batch_size=2, verbose=1))  # 2
                    # del inp

                    cur_pred = model_out if save else ensemble_prediction
                    mc_pred = np.zeros((actual_batch_size * 2, DIM[0], DIM[1], DIM[2], NUM_CLASS))

                    del model_out

                    # Z = αZ + (1 - α)z
                    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred
                    save_array_kits(self.ensemble_path, ensemble_prediction, 'segm', start, end)
                    del ensemble_prediction

                    # mc dropout chnages
                    T = DROPOUT_NUM
                    for i in np.arange(T):
                        print(b_no, i)
                        model_out = p_model_MC.predict(inp, batch_size=2, verbose=1)
                        mc_pred = np.add(model_out, mc_pred)

                    # avg_pred = mc_pred / T#
                    entropy = -((mc_pred / T) * np.log((mc_pred / T) + 1e-5))
                    del mc_pred, inp, model_out

                    max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))
                    indices = np.argpartition(max_pred_ravel, -confident_pixels_no)[-confident_pixels_no:]

                    mask = np.ones(max_pred_ravel.shape, dtype=bool)
                    mask[indices] = False

                    max_pred_ravel[mask] = 0
                    max_pred_ravel = np.where(max_pred_ravel > 0, np.ones_like(max_pred_ravel) * 2,
                                              np.zeros_like(max_pred_ravel))
                    flag = np.reshape(max_pred_ravel, (actual_batch_size * 2, DIM[0], DIM[1], DIM[2], 1))
                    del max_pred_ravel, indices

                    save_array_kits(self.ensemble_path, flag, 'flag', start, end)

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
                                                   monitor='val_dice_coef',
                                                   save_best_only=True,
                                                   verbose=1,
                                                   mode='max')
    else:
        model_checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_dice_coef',
                                           save_best_only=True,
                                           verbose=1,
                                           mode='max')

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=True, histogram_freq=0,
                              batch_size=2, write_images=False)

    train_id_list = []
    for i in np.arange(num_train_data):
        train_id_list.append(str(i) + '#right')
        train_id_list.append(str(i) + '#left')

    print(train_id_list[0:10])

    np.random.shuffle(train_id_list)
    tcb = TemporalCallback(DATA_PATH, ENS_GT_PATH, train_id_list)
    lcb = wm.LossCallback()
    es = EarlyStopping(monitor='val_dice_coef', mode='max', verbose=1, patience=100)
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger, es]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    # params = {'dim': (32, 168, 168),'batch_size': batch_size}

    print('-' * 30)
    print('Fitting training_scripts...')
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

    val_fold = np.load('/data/suhita/temporal/kits/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data * 2, DIM[0], DIM[1], DIM[2], 1), dtype='int8') * 3
    val_img_arr = np.zeros((num_val_data * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((num_val_data * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    VAL_DATA = '/data/suhita/temporal/kits/preprocessed_labeled_train'
    for i in range(num_val_data):
        val_img_arr[i * 2, :, :, :, 0] = np.load(os.path.join(VAL_DATA, val_fold[i], 'img_left.npy'))
        val_img_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(VAL_DATA, val_fold[i], 'img_right.npy'))
        val_GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(VAL_DATA, val_fold[i], 'segm_left.npy'))
        val_GT_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(VAL_DATA, val_fold[i], 'segm_right.npy'))

    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr]
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    # workers=4)
    # training_scripts.save('temporal_max_ramp_final.h5')


def predict(model_name):
    data_path = '/data/suhita/temporal/kits/preprocessed_labeled_train'

    val_fold = np.load('/data/suhita/temporal/kits/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data * 2, DIM[0], DIM[1], DIM[2], 1), dtype='int8')
    img_arr = np.zeros((val_fold.shape[0] * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    GT_arr = np.zeros((val_fold.shape[0] * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)

    for i in range(val_fold.shape[0]):
        img_arr[i * 2, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'img_left.npy'))
        img_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'img_right.npy'))
        GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_left.npy'))
        GT_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_right.npy'))

    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=model_name, )
    # model.load_weights(model_name)

    # single image evaluation
    # for i in range(0,val_fold.shape[0]*2):
    #   out_eval = model.evaluate([img_arr[i:i+1],GT_arr[i:i+1],val_supervised_flag[i:i+1]], GT_arr[i:i+1], batch_size=1, verbose=0)
    #  print(val_fold[int(i/2)],out_eval)

    out_eval = model[0].evaluate([img_arr, GT_arr, val_supervised_flag], GT_arr, batch_size=2, verbose=1)
    print(out_eval)


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
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.compat.v1.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    # train(gpu, nb_gpus)
    train(None, None)
    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/cache/suhita/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = np.load('/cache/suhita/data/final_test_array_imgs.npy')
    val_y = np.load('/cache/suhita/data/final_test_array_GT.npy').astype('int8')
    # model_name = '/data/suhita/temporal/kits_softmax_F1_150_Perct_UL50_Labelled_P_0.5.h5'
    predict(MODEL_NAME)
