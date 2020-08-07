import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from old.utils.AugmentationGenerator import *
from prostate.config import *
from prostate.generator.temporal_A import DataGenerator as train_gen
from prostate.model.bai import weighted_model
from utility.callbacks.bai import TemporalCallback
from utility.constants import *
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel
from utility.prostate.utils import get_val_data
from utility.utils import cleanup


def train(gpu_id, nb_gpus, ens_path, labelled_percentage, fold_num, name, lr=LR):
    DATA_PATH = PROSTATE_DATA_ROOT + 'fold_' + str(fold_num) + '_P' + str(labelled_percentage) + '/train/'
    TB_LOG_DIR = SAVE_PATH + '/tb/prostate/' + name + '_' + str(lr) + '/'
    MODEL_NAME = SAVE_PATH + '/model/prostate/' + name + H5
    CSV_NAME = SAVE_PATH + '/csv/prostate/' + name + '.csv'
    TRAINED_MODEL_PATH = TRAINED_MODEL_ROOT_PATH + '/prostate/supervised_F' + str(fold_num) + '_P' + str(
        labelled_percentage) + '.h5'

    num_labeled_train = int(labelled_percentage * PROSTATE_LABELLED_TRAIN_NUM)
    num_train_data = len(os.listdir(os.path.join(DATA_PATH, IMGS)))
    num_un_labeled_train = num_train_data - num_labeled_train

    print('-' * 30)
    print('Loading train data...')
    print('-' * 30)

    # Build Model
    wm = weighted_model()
    model = wm.build_model(TRAINED_MODEL_PATH, img_shape=(PROSTATE_DIM[0], PROSTATE_DIM[1], PROSTATE_DIM[2]),
                           learning_rate=lr,
                           gpu_id=gpu_id,
                           nb_gpus=nb_gpus)

    print("Images Size:", num_train_data)
    print("Unlabeled Size:", num_un_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model.summary()

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

    tcb = TemporalCallback(PROSTATE_DIM, DATA_PATH, ens_path, SAVE_PATH, num_train_data, num_labeled_train,
                           PATIENTS_PER_BATCH, PROSTATE_NR_CLASS, BATCH_SIZE, PROSTATE_DATASET)
    lcb = wm.LossCallback()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.001)
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger, es]

    print('BATCH Size = ', BATCH_SIZE)

    print('Callbacks: ', cb)
    # params = {'dim': (32, 168, 168),'batch_size': batch_size}

    print('-' * 30)
    print('Fitting training_scripts...')
    print('-' * 30)
    training_generator = train_gen(DATA_PATH,
                                   ens_path,
                                   train_id_list,
                                   batch_size=BATCH_SIZE,
                                   labelled_num=num_labeled_train)

    # steps = num_train_data / batch_size
    steps = (num_train_data * AUGMENTATION_NO) / BATCH_SIZE
    # steps = 2

    x_val, y_val = get_val_data(DATA_PATH)
    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=NUM_EPOCH,
                                  callbacks=cb
                                  )


if __name__ == '__main__':
    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # train(gpu, nb_gpus)
    try:
        fold_num = 1
        perc = 1.
        ens_path = TEMP_PATH + 'sadv2/'
        name = 'test_bai_F' + str(fold_num) + '_Perct_Labelled_' + str(perc)
        train(None, None, ens_path=ens_path, labelled_percentage=perc, fold_num=fold_num, name=name)

    finally:

        if os.path.exists(ens_path):
            cleanup()
        print('clean up done!')

    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
