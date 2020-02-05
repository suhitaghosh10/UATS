import sys
sys.path.append('../')

import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from kits.data_generation import DataGenerator as train_gen
from kits.model import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_complete_array
from kits import utils
import SimpleITK as sitk

data_path = '/data/suhita/temporal/kits/preprocessed_labeled_train'
out_dir = '/data/suhita/temporal/kits/'





learning_rate = 5e-5
AUGMENTATION_NO = 5
TRAIN_NUM = 50
#PERC = 0.25
FOLD_NUM = 1




NUM_CLASS = 1
num_epoch = 1000
batch_size = 2
DIM = [152,152,56]


def remove_tumor_segmentation(arr):
    arr[arr > 1] = 1
    return arr


def train(gpu_id, nb_gpus, trained_model=None, perc=1.0, augmentation = False):

    if augmentation:
        augm = '_augm'
    else:
        augm = ''

    NAME = '1_supervised_F_centered_BB_' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_' + str(
        learning_rate) + '_Perc_' + str(perc)+augm
    CSV_NAME = out_dir + NAME + '.csv'

    TB_LOG_DIR = '/data/suhita/temporal/tb/kits/' + NAME + '_' + str(learning_rate) + '/'
    MODEL_NAME = out_dir + NAME + '.h5'
    TRAINED_MODEL_PATH = MODEL_NAME

    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0],DIM[1],DIM[2]), learning_rate=learning_rate)

    print('-' * 30)
    print('Creating and compiling model_impl...')
    print('-' * 30)

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

    tensorboard = TensorBoard(log_dir=TB_LOG_DIR, write_graph=False, write_grads=False, histogram_freq=0, write_images=False)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # datagen listmake_dataset
    train_fold = np.load('/data/suhita/temporal/kits/Folds/train_fold' + str(FOLD_NUM) + '.npy')
    print(train_fold[0:10])
    nr_samples = train_fold.shape[0]

    # np.random.seed(5)
    np.random.seed(1234)
    np.random.shuffle(train_fold)
    print(train_fold[0:10])

    train_fold = train_fold[:int(nr_samples*perc)]

    train_id_list = []
    for i in range(train_fold.shape[0]):
        train_id_list.append(os.path.join(train_fold[i], 'img_left.npy'))
        train_id_list.append(os.path.join(train_fold[i], 'img_right.npy'))


    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag

    cb = [model_checkpoint, tensorboard, es, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (DIM[0],DIM[1],DIM[2]),
              'batch_size': batch_size }

    print('-' * 30)
    print('Fitting model_impl...')
    print('-' * 30)



    training_generator = train_gen(data_path,
                                   train_id_list,
                                   augmentation = augmentation,
                                   **params)

    val_fold = np.load('/data/suhita/temporal/kits/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    # val_id_list = []
    # for i in range(val_fold.shape[0]):
    #     val_id_list.append(os.path.join(val_fold[i], 'img_left.npy'))
    #     val_id_list.append(os.path.join(val_fold[i], 'img_right.npy'))
    #
    # val_generator = train_gen(data_path,
    #                                val_id_list,
    #                                **params)

    val_img_arr = np.zeros((val_fold.shape[0]*2, DIM[0],DIM[1],DIM[2],1), dtype = float)
    val_GT_arr = np.zeros((val_fold.shape[0] * 2, DIM[0],DIM[1],DIM[2], 1), dtype=float)

    for i in range(val_fold.shape[0]):
        val_img_arr[i*2,:,:,:,0] = np.load(os.path.join(data_path, val_fold[i], 'img_left.npy'))
        val_img_arr[i * 2 +1,:,:,:,0] = np.load(os.path.join(data_path, val_fold[i], 'img_right.npy'))
        val_GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_left.npy'))
        val_GT_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_right.npy'))



    if augmentation == False:
        augm_no = 1
    else:
        augm_no = AUGMENTATION_NO
    steps = (TRAIN_NUM * augm_no) / batch_size
    steps = 2

    # get validation fold
    #val_fold = np.load('Folds/val_fold'+str(FOLD_NUM)+'.npy')
    # val_x_arr = get_complete_array(data_path, val_fold, GT = False)
    # val_y_arr = get_complete_array(data_path, val_fold, dtype='int8', GT = True)



    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[val_img_arr, val_GT_arr],
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    del val_GT_arr, val_img_arr

    # workers=4)
    # model_impl.save('temporal_max_ramp_final.h5')


def generate_prediction_for_ul(model_name, onlyEval=False, img_path=None, ens_path=None):
    pred_dir = out_dir + '/predictions/'

    # val_fold = np.load(out_dir+'/Folds/val_fold' + str(FOLD_NUM) + '.npy')
    val_fold = os.listdir(img_path)

    img_arr = np.zeros((len(val_fold) * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)
    GT_arr = np.zeros((len(val_fold) * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)

    for i in range(len(val_fold)):
        img_arr[i * 2, :, :, :, 0] = np.load(os.path.join(img_path, val_fold[i], 'img_left.npy'))
        img_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(img_path, val_fold[i], 'img_right.npy'))
        GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(img_path, val_fold[i], 'segm_left.npy'))
        GT_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(img_path, val_fold[i], 'segm_right.npy'))

    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0], DIM[1], DIM[2]), learning_rate=learning_rate)
    model.load_weights(model_name)

    if onlyEval:
        out_value = model.evaluate(img_arr, GT_arr, batch_size=1, verbose=0)
        print(out_value)
    else:
        out = model.predict(img_arr, batch_size=2, verbose=1)
        # np.save(os.path.join(out_dir, 'predicted.npy'), out)
        for i in range(out.shape[0]):
            # segm = sitk.GetImageFromArray(out[i,:,:,:,0])
            # utils.makeDirectory(os.path.join(pred_dir, val_fold[int(i/2)]))
            utils.makeDirectory(os.path.join(ens_path, val_fold[int(i / 2)]))
            if i % 2 == 0:
                # img = sitk.ReadImage(os.path.join(data_path, val_fold[int(i / 2)], 'img_left.nrrd'))
                # segm.CopyInformation(img)
                # sitk.WriteImage(img, os.path.join(pred_dir, val_fold[int(i / 2)], 'img_left.nrrd'))

                np.save(os.path.join(ens_path, val_fold[int(i / 2)], 'img_left.npy'),
                        np.load(os.path.join(img_path, val_fold[int(i / 2)], 'img_left.npy')))
                # sitk.WriteImage(segm, os.path.join(pred_dir, val_fold[int(i/2)], 'segm_left.nrrd'))
                np.save(os.path.join(ens_path, val_fold[int(i / 2)], 'segm_left.npy'), out[i, :, :, :, 0])

            else:
                # img = sitk.ReadImage(os.path.join(data_path, val_fold[int(i / 2)], 'img_right.nrrd'))
                # segm.CopyInformation(img)
                # sitk.WriteImage(img, os.path.join(pred_dir, val_fold[int(i / 2)], 'img_right.nrrd'))
                np.save(os.path.join(ens_path, val_fold[int(i / 2)], 'img_right.npy'),
                        np.load(os.path.join(img_path, val_fold[int(i / 2)], 'img_right.npy')))
                # sitk.WriteImage(segm, os.path.join(pred_dir, val_fold[int(i/2)], 'segm_right.nrrd'))
                np.save(os.path.join(ens_path, val_fold[int(i / 2)], 'segm_right.npy'), out[i, :, :, :, 0])

        # single image evaluation
        # for i in range(0,val_fold.shape[0]*2):
        #    out_eval = model.evaluate(img_arr[i:i+1], GT_arr[i:i+1], batch_size=1, verbose=0)
        #   print(val_fold[int(i/2)],out_eval)

    # if eval:
    #     val_gt = val_gt.astype(np.uint8)
    #     val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],
    #                    val_gt[:, :, :, :, 4]]
    #     scores = model.evaluate([val_imgs], val_gt_list, batch_size=2, verbose=1)
    #     length = len(model.metrics_names)
    #     for i in range(6, 11):
    #         print("%s: %.16f%%" % (model.metrics_names[i], scores[i]))


def predict(model_name, onlyEval=False):
    pred_dir = '/home/anneke/projects/uats/code/kits/output/predictions/'

    val_fold = np.load('/data/suhita/temporal/kits/Folds/val_fold' + str(FOLD_NUM) + '.npy')

    img_arr = np.zeros((val_fold.shape[0]*2, DIM[0],DIM[1],DIM[2],1), dtype = float)
    GT_arr = np.zeros((val_fold.shape[0] * 2, DIM[0], DIM[1], DIM[2], 1), dtype=float)

    for i in range(val_fold.shape[0]):
        img_arr[i*2,:,:,:,0] = np.load(os.path.join(data_path, val_fold[i], 'img_left.npy'))
        img_arr[i * 2 +1,:,:,:,0] = np.load(os.path.join(data_path, val_fold[i], 'img_right.npy'))
        GT_arr[i * 2, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_left.npy'))
        GT_arr[i * 2 + 1, :, :, :, 0] = np.load(os.path.join(data_path, val_fold[i], 'segm_right.npy'))


    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0],DIM[1],DIM[2]), learning_rate=learning_rate)
    model.load_weights(model_name)

    if onlyEval:
        out_value = model.evaluate(img_arr, GT_arr, batch_size=1, verbose=0)
        print(out_value)
    else:
        out = model.predict(img_arr, batch_size=2, verbose=1)
        #np.save(os.path.join(out_dir, 'predicted.npy'), out)
        for i in range(out.shape[0]):
            segm = sitk.GetImageFromArray(out[i,:,:,:,0])
            utils.makeDirectory(os.path.join(pred_dir, val_fold[int(i/2)]))
            if i%2 ==0:
                img = sitk.ReadImage(os.path.join(data_path, val_fold[int(i / 2)], 'img_left.nrrd'))
                segm.CopyInformation(img)
                sitk.WriteImage(img, os.path.join(pred_dir, val_fold[int(i / 2)], 'img_left.nrrd'))
                sitk.WriteImage(segm, os.path.join(pred_dir, val_fold[int(i/2)], 'segm_left.nrrd'))

            else:
                img = sitk.ReadImage(os.path.join(data_path, val_fold[int(i / 2)], 'img_right.nrrd'))
                segm.CopyInformation(img)
                sitk.WriteImage(img, os.path.join(pred_dir, val_fold[int(i / 2)], 'img_right.nrrd'))
                sitk.WriteImage(segm, os.path.join(pred_dir, val_fold[int(i/2)], 'segm_right.nrrd'))

        # single image evaluation
        for i in range(0,val_fold.shape[0]*2):
            out_eval = model.evaluate(img_arr[i:i+1], GT_arr[i:i+1], batch_size=1, verbose=0)
            print(val_fold[int(i/2)],out_eval)

if __name__ == '__main__':


    gpu = '/GPU:0'
    batch_size = batch_size

    gpu_id = '1'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # config = tf.ConfigProto()
    #
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # set_session(tf.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)



    # perc = 0.25
    # train(None, None, perc=perc, augmentation=True)
    # #train(None, None, perc=perc, augmentation=False)

    model_name = '/data/suhita/temporal/kits/models/1_supervised_Perc_0.5.h5'
    img_path = '/cache/suhita/data/kidney_anneke/preprocessed_unlabeled/'
    # generate_prediction_for_ul(model_name, onlyEval=False, img_path=img_path, ens_path=out_dir+'/output/UL_1.0')

    predict(model_name, onlyEval=True)
