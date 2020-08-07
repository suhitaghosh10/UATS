import shutil
from shutil import copyfile

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from hippocampus import DataGenerator as train_gen
from hippocampus.model_entropy import weighted_model
from hippocampus.utils import get_multi_class_arr
from kits.utils import makedir
from old.preprocess_images import get_array, save_array
from old.utils.AugmentationGenerator import *
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

learning_rate = 4e-5
AUGMENTATION_NO = 5
augmentation = True
PERCENTAGE_OF_PIXELS = 50
FOLD_NUM = 3

num_epoch = 1000
batch_size = 4
DIM = [48, 64, 48]

ramp_up_period = 50
ramp_down_period = 50
alpha = 0.6

ENS_GT_PATH = '/data/suhita/temporal/hippocampus/output/sad000003/'


def train(gpu_id, nb_gpus, perc):
    PERCENTAGE_OF_LABELLED = perc
    DATA_PATH = '/cache/suhita/data/hippocampus/fold_' + str(FOLD_NUM) + '_P' + str(PERCENTAGE_OF_LABELLED) + '/'
    TRAIN_NUM = len(np.load('/cache/suhita/hippocampus/Folds/train_fold' + str(FOLD_NUM) + '.npy'))
    NAME = '2_hippocampus_T_20_mc_F' + str(FOLD_NUM) + '_Perct_Labelled_' + str(PERCENTAGE_OF_LABELLED)
    TB_LOG_DIR = '/data/suhita/temporal/tb/hippocampus/' + NAME + '_' + str(learning_rate) + '/'
    MODEL_NAME = '/data/suhita/temporal/hippocampus/' + NAME + '.h5'

    CSV_NAME = '/data/suhita/temporal/CSV/' + NAME + '.csv'
    TRAINED_MODEL_PATH = '/data/suhita/hippocampus/output/models/2_supervised_F_' + str(
        FOLD_NUM) + '_150_4e-05_Perc_' + str(
        PERCENTAGE_OF_LABELLED) + '_augm.h5'
    # TRAINED_MODEL_PATH = MODEL_NAME
    num_labeled_train = int(PERCENTAGE_OF_LABELLED * TRAIN_NUM)
    num_train_data = len(os.listdir(DATA_PATH + '/imgs/'))
    num_un_labeled_train = num_train_data - num_labeled_train
    IMGS_PER_ENS_BATCH = num_un_labeled_train // 2

    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()

    model, model_MC = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate, gpu_id=gpu_id,
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
            self.val_z1_dice_coef = 0.
            self.val_z2_dice_coef = 0.

            self.data_path = data_path
            self.ensemble_path = ensemble_path
            self.train_idx_list = train_idx_list

            flag = np.ones(shape=DIM, dtype='float16')
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
                            np.zeros(shape=DIM, dtype='float16'))

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
            model_MC.set_weights(model.get_weights())
            # print(time() - self.starttime)
            # model_temp = model

            bg_save, self.val_bg_dice_coef = self.shall_save(logs['val_bg_dice_coef'], self.val_bg_dice_coef)
            z1_save, self.val_z1_dice_coef = self.shall_save(logs['val_z1_dice_coef'], self.val_z1_dice_coef)
            z2_save, self.val_z2_dice_coef = self.shall_save(logs['val_z2_dice_coef'], self.val_z2_dice_coef)

            if epoch > 0:

                patients_per_batch = IMGS_PER_ENS_BATCH
                num_batches = num_un_labeled_train // patients_per_batch
                remainder = num_un_labeled_train % patients_per_batch
                remainder_pixels = remainder * DIM[0] * DIM[1] * DIM[2]
                confident_pixels_no_per_batch = (PERCENTAGE_OF_PIXELS * patients_per_batch * DIM[0] * DIM[1] * DIM[
                    2]) // 100
                if remainder_pixels < confident_pixels_no_per_batch:
                    patients_per_last_batch = patients_per_batch + remainder
                else:
                    patients_per_last_batch = remainder
                    num_batches = num_batches + 1

                for b_no in np.arange(num_batches):
                    actual_batch_size = patients_per_batch if (
                            b_no < num_batches - 1) else patients_per_last_batch
                    confident_pixels_no = (PERCENTAGE_OF_PIXELS * DIM[0] * DIM[1] * DIM[2] * actual_batch_size) // 100
                    start = (b_no * patients_per_batch) + num_labeled_train
                    end = (start + actual_batch_size)
                    imgs = get_array(self.data_path + '/imgs/', start, end)
                    ensemble_prediction = get_array(self.ensemble_path + '/ens_gt/', start, end, dtype='float32')
                    supervised_flag = get_array(self.ensemble_path + '/flag/', start, end, dtype='float16')

                    inp = [imgs, ensemble_prediction, supervised_flag]
                    del imgs, supervised_flag

                    cur_pred = np.zeros((actual_batch_size, DIM[0], DIM[1], DIM[2], 3))
                    model_out = model.predict(inp, batch_size=batch_size, verbose=1)  # 1

                    cur_pred[:, :, :, :, 0] = model_out[0] if bg_save else ensemble_prediction[:, :, :, :, 0]
                    cur_pred[:, :, :, :, 1] = model_out[1] if z1_save else ensemble_prediction[:, :, :, :, 1]
                    cur_pred[:, :, :, :, 2] = model_out[2] if z2_save else ensemble_prediction[:, :, :, :, 2]

                    del model_out

                    # Z = αZ + (1 - α)z
                    ensemble_prediction = alpha * ensemble_prediction + (1 - alpha) * cur_pred
                    save_array(os.path.join(self.ensemble_path, 'ens_gt'), ensemble_prediction, start, end)
                    del ensemble_prediction

                    # mc dropout chnages
                    mc_pred = np.zeros((actual_batch_size, DIM[0], DIM[1], DIM[2], 3))
                    T = 20
                    for i in np.arange(T):
                        model_out = model_MC.predict(inp, batch_size=batch_size, verbose=1)

                        mc_pred[:, :, :, :, 0] = np.add(model_out[0], mc_pred[:, :, :, :, 0])
                        mc_pred[:, :, :, :, 1] = np.add(model_out[1], mc_pred[:, :, :, :, 1])
                        mc_pred[:, :, :, :, 2] = np.add(model_out[2], mc_pred[:, :, :, :, 2])

                    # avg_pred = mc_pred / T#
                    entropy = None
                    for z in np.arange(3):
                        if z == 0:
                            entropy = (mc_pred[:, :, :, :, z] / T) * np.log((mc_pred[:, :, :, :, z] / T) + 1e-5)
                        else:
                            entropy = entropy + (mc_pred[:, :, :, :, z] / T) * np.log(
                                (mc_pred[:, :, :, :, z] / T) + 1e-5)
                    entropy = -entropy
                    del mc_pred, inp, model_out

                    argmax_pred_ravel = np.ravel(np.argmin(cur_pred, axis=-1))
                    max_pred_ravel = np.ravel(np.max(cur_pred, axis=-1))

                    indices = None
                    del cur_pred
                    for zone in np.arange(3):
                        entropy_zone = np.ravel(entropy[:, :, :, :])
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
                    flag = np.reshape(max_pred_ravel, (actual_batch_size, DIM[0], DIM[1], DIM[2]))
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
    np.random.shuffle(train_id_list)

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

    val_fold = np.load('/cache/suhita/hippocampus/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    num_val_data = len(val_fold)
    val_supervised_flag = np.ones((num_val_data, DIM[0], DIM[1], DIM[2]), dtype='int8')
    val_img_arr = np.zeros((num_val_data, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((num_val_data, DIM[0], DIM[1], DIM[2], 3), dtype=float)

    for i in np.arange(num_val_data):
        val_img_arr[i, :, :, :, 0] = np.load(os.path.join('/cache/suhita/hippocampus/preprocessed/labelled/train',
                                                          val_fold[i].replace('.nii.gz', '.npy')))
        val_GT_arr[i] = get_multi_class_arr(np.load(
            os.path.join('/cache/suhita/hippocampus/preprocessed/labelled-GT/train', val_fold[i]).replace('.nii.gz',
                                                                                                          '.npy')), 3)

    x_val = [val_img_arr, val_GT_arr, val_supervised_flag]
    y_val = [val_GT_arr[:, :, :, :, 0], val_GT_arr[:, :, :, :, 1], val_GT_arr[:, :, :, :, 2]]
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
    val_supervised_flag = np.ones((num_val_data, DIM[0], DIM[1], DIM[2], 1), dtype='int8')
    img_arr = np.zeros((val_fold.shape[0], DIM[0], DIM[1], DIM[2], 1), dtype=float)
    GT_arr = np.zeros((val_fold.shape[0], DIM[0], DIM[1], DIM[2], 1), dtype=float)

    for i in range(val_fold.shape[0]):
        img_arr[i, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'img_left.npy'))
        GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_left.npy'))

    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), num_class=1, use_dice_cl=False,
                           learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=model_name, temp=1)
    model.load_weights(model_name)

    # single image evaluation
    # for i in range(0,val_fold.shape[0]*2):
    #   out_eval = model.evaluate([img_arr[i:i+1],GT_arr[i:i+1],val_supervised_flag[i:i+1]], GT_arr[i:i+1], batch_size=1, verbose=0)
    #  print(val_fold[int(i/2)],out_eval)

    out_eval = model.evaluate([img_arr, GT_arr, val_supervised_flag], GT_arr, batch_size=2, verbose=0)
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
    try:
        train(None, None, 0.1)
        # shutil.rmtree(ENS_GT_PATH)
        # train(None, None, 0.25)
        # shutil.rmtree(ENS_GT_PATH)
        # train(None, None, 0.25)
        # shutil.rmtree(ENS_GT_PATH)
        #train(None, None, 0.05)

    finally:

        if os.path.exists(ENS_GT_PATH):
            shutil.rmtree(ENS_GT_PATH)
        print('clean up done!')

    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
