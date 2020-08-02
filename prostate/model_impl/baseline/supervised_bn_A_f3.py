import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from generator.baseline_A import DataGenerator as train_gen
from lib.segmentation.model.model_baseline import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_complete_array
from keras.backend import clear_session

learning_rate = 5e-5
AUGMENTATION_NO = 5
TRAIN_NUM = 58

TRAIN_IMGS_PATH = '/cache/suhita/data/prostate/fold_3/train/imgs/'
TRAIN_GT_PATH = '/cache/suhita/data/prostate/fold_3/train/gt/'

VAL_IMGS_PATH = '/cache/suhita/data/prostate/fold_3/val/imgs/'
VAL_GT_PATH = '/cache/suhita/data/prostate/fold_3/val/gt/'

NUM_CLASS = 5
num_epoch = 1000
batch_size = 2


def train(gpu_id, nb_gpus, trained_model=None, perc=None, fold=None):
    NAME = 'supervised_F' + str(fold) + '_P' + str(perc)
    CSV_NAME = '/data/suhita/temporal/CSV/' + NAME + '.csv'
    TB_LOG_DIR = '/data/suhita/temporal/tb/prostate/' + NAME + str(learning_rate) + '/'
    MODEL_NAME = '/data/suhita/prostate/' + NAME + '.h5'

    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=trained_model)

    print('-' * 30)
    print('Creating and compiling model_impl...')
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
                              batch_size=2, write_images=False)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # datagen listmake_dataset
    num_labeled_train = int(perc * TRAIN_NUM)
    train_id_list = [str(i) for i in np.arange(TRAIN_NUM)]
    np.random.seed(1234)
    np.random.shuffle(train_id_list)
    train_id_list = train_id_list[0:num_labeled_train]
    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag

    cb = [model_checkpoint, tensorboard, es, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting model_impl...')
    print('-' * 30)
    training_generator = train_gen(TRAIN_IMGS_PATH,
                                   TRAIN_GT_PATH,
                                   train_id_list,
                                   **params)

    steps = (num_labeled_train * AUGMENTATION_NO) / batch_size
    # steps=2

    val_x_arr = get_complete_array(VAL_IMGS_PATH)
    val_y_arr = get_complete_array(VAL_GT_PATH, dtype='int8')

    pz = val_y_arr[:, :, :, :, 0]
    cz = val_y_arr[:, :, :, :, 1]
    us = val_y_arr[:, :, :, :, 2]
    afs = val_y_arr[:, :, :, :, 3]
    bg = val_y_arr[:, :, :, :, 4]

    y_val = [pz, cz, us, afs, bg]
    x_val = [val_x_arr]
    del pz, cz, us, afs, bg, val_y_arr, val_x_arr

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )
    del model

    # workers=4)
    # model_impl.save('temporal_max_ramp_final.h5')


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = 2
    gpu_id = '1'
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

    train(None, None, trained_model=None, perc=1.0, fold=3)
    # clear_session()
    # train(None, None, trained_model=None, perc=0.5)
    # train(None, None, trained_model=None, perc=1.0)
    # train(gpu, nb_gpus, trained_model=None)
    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/cache/suhita/data/validation/valArray_GT_fold1.npy').astype('int8')

    # val_x = TEST_IMGS_PATH
    # val_y = TEST_GT_PATH
    # predict(val_x, val_y, TRAINED_MODEL_PATH)
