import sys
sys.path.append('../')

import os

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from hippocampus.data_generation import DataGenerator as train_gen
from hippocampus.model import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel

#
data_path = '/cache/suhita/hippocampus/preprocessed/labelled/train'
data_path_GT = '/cache/suhita/hippocampus/preprocessed/labelled-GT/train'
out_dir = '/data/suhita/hippocampus/output/models/'
#




learning_rate = 4e-5
AUGMENTATION_NO = 5
TRAIN_NUM = 150
# PERC = 0.25
FOLD_NUM = 1

#NUM_CLASS = 1
num_epoch = 1000
batch_size = 4
DIM = [48, 64, 48]


def get_multi_class_arr(arr, n_classes=3):
    size = arr.shape
    out_arr = np.zeros([size[0], size[1], size[2], n_classes])

    for i in range(n_classes):
        arr_temp = arr.copy()
        out_arr[:, :, :, i] = np.where(arr_temp == i, 1, 0)
        del arr_temp
    return out_arr


def train(gpu_id, nb_gpus, trained_model=None, perc=1.0, augmentation=False):

    if augmentation:
        augm = '_augm'
    else:
        augm = ''

    NAME = '2_supervised_F_' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_' + str(
        learning_rate) + '_Perc_' + str(perc) + augm
    CSV_NAME = out_dir + NAME + '.csv'

    TB_LOG_DIR = out_dir + NAME + str(learning_rate) + '/'
    MODEL_NAME = out_dir + NAME + '.h5'

    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate)

    print('-' * 30)
    print('Creating and compiling model_impl...')
    print('-' * 30)
    print(model.summary())

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

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=False, histogram_freq=0,
                              write_images=False)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.0005)

    # datagen listmake_dataset
    train_fold = np.load('/cache/suhita/hippocampus/Folds/train_fold' + str(FOLD_NUM) + '.npy')
    nr_samples = train_fold.shape[0]

    # np.random.seed(5)
    np.random.seed(1234)
    np.random.shuffle(train_fold)
    print(train_fold[0:10])

    train_fold = train_fold[:int(nr_samples * perc)]

    train_id_list = []
    for i in range(train_fold.shape[0]):
        train_id_list.append(train_fold[i][:-7] + '.npy')


    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag

    cb = [model_checkpoint, tensorboard, es, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (DIM[0], DIM[1], DIM[2]),
              'batch_size': batch_size,
              'GT_path': data_path_GT,
              'n_classes': 3}

    print('-' * 30)
    print('Fitting model_impl...')
    print('-' * 30)

    training_generator = train_gen(data_path,
                                   train_id_list,
                                   augmentation=augmentation,
                                   **params)

    val_fold = np.load('/cache/suhita/hippocampus/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    # val_id_list = []
    # for i in range(val_fold.shape[0]):
    #     val_id_list.append(os.path.join(val_fold[i], 'img_left.npy'))
    #     val_id_list.append(os.path.join(val_fold[i], 'img_right.npy'))
    #
    # val_generator = train_gen(data_path,
    #                                val_id_list,
    #                                **params)

    val_img_arr = np.zeros((val_fold.shape[0], DIM[0], DIM[1], DIM[2], 1), dtype=float)
    val_GT_arr = np.zeros((val_fold.shape[0], DIM[0], DIM[1], DIM[2], 3), dtype=np.uint8)

    for i in range(val_fold.shape[0]):
        val_img_arr[i, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i][:-7] + '.npy'))
        val_GT = np.load(os.path.join(data_path_GT, val_fold[i][:-7] + '.npy'))
        val_GT = get_multi_class_arr(val_GT, n_classes=3)
        val_GT_arr[i] = val_GT

    if augmentation == False:
        augm_no = 1
    else:
        augm_no = AUGMENTATION_NO
    steps = (TRAIN_NUM) / batch_size
    # steps=2

    # get validation fold
    #val_fold = np.load('Folds/val_fold'+str(FOLD_NUM)+'.npy')
    # val_x_arr = get_complete_array(data_path, val_fold, GT = False)
    # val_y_arr = get_complete_array(data_path, val_fold, dtype='int8', GT = True)

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[val_img_arr, [val_GT_arr[:, :, :, :, 0],
                                                                 val_GT_arr[:, :, :, :, 1],
                                                                 val_GT_arr[:, :, :, :, 2]]],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    del val_GT_arr, val_img_arr

    # workers=4)
    # model_impl.save('temporal_max_ramp_final.h5')

if __name__ == '__main__':
    GPU_ID = '2'
    # gpu = '/GPU:0'
    gpu = '/GPU:0'
    batch_size = batch_size

    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # config = tf.ConfigProto()
    #
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # set_session(tf.Session(config=config))

    nb_gpus = len(GPU_ID.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    # perc = 0.05
    # train(None, None, perc=perc, augmentation=True)
    #
    # perc = 0.1
    #train(None, None, perc=perc, augmentation=True)

    # perc = 0.25
    #train(None, None, perc=perc, augmentation=True)

    perc = 1.0
    train(None, None, perc=perc, augmentation=True)

    # predict(out_dir+'/supervised_F_centered_BB_1_50_0.0005_Perc_0.5_augm.h5', onlyEval=True)

    #predict_unlabeled('/home/anneke/projects/uats/code/kits/output/models/supervised_F_centered_BB_1_50_5e-05_Perc_1.0_augm.h5')
