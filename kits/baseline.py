import sys
sys.path.append('../')

from kits import preprocess

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

data_path = '/cache/anneke/kits_challenge/kits19/data/preprocessed_labeled_train'
out_dir = '/home/anneke/projects/uats/code/kits/output/models/'





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

    TB_LOG_DIR = out_dir + NAME + str(learning_rate) + '/'
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
    train_fold = np.load('Folds/train_fold'+str(FOLD_NUM)+'.npy')
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


    val_fold = np.load('Folds/val_fold'+str(FOLD_NUM)+'.npy')
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
    # steps=2

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



def predict_unlabeled(model_name, pred_dir = '/home/anneke/projects/uats/code/kits/output/predictions/'):


    DIM = [80,200,200]
    SPACING = [4.0, 1.0, 1.0]


    img_dir = '/data/anneke/kits_challenge/kits19/data/preprocessed_unlabeled'
    cases = sorted(os.listdir(img_dir))

    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[2], DIM[1], DIM[0]), learning_rate=learning_rate)
    model.load_weights(model_name)

    for i in range(2):#len(cases)):
        img_arr = np.zeros([2, DIM[2], DIM[1], DIM[0],1])


        img_l = sitk.ReadImage(os.path.join(img_dir, cases[i], 'img_left.nrrd'))
        img_r = sitk.ReadImage(os.path.join(img_dir, cases[i], 'img_right.nrrd'))

        img_l, img_r = preprocess.normalizeIntensities(img_l, img_r)


        img_arr[0,:,:,:,0] = sitk.GetArrayFromImage(img_l)
        img_arr[1,:,:,:,0] = sitk.GetArrayFromImage(img_r)

        out_arr = model.predict(img_arr, batch_size=2 )
        pred_l = sitk.GetImageFromArray(out_arr[0,:,:,:,0])
        pred_r = sitk.GetImageFromArray(out_arr[1, :, :, :, 0])

        pred_l.CopyInformation(img_l)
        pred_r.CopyInformation(img_r)

        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkUInt8)

        pred_l = castImageFilter.Execute(pred_l)
        del castImageFilter
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
        pred_r = castImageFilter.Execute(pred_r)

        pred_l= utils.getLargestConnectedComponents(pred_l)
        pred_r = utils.getLargestConnectedComponents(pred_r)



        utils.makeDirectory(pred_dir+'/'+cases[i])

        sitk.WriteImage(pred_r, os.path.join(pred_dir, cases[i], 'segm_right.nrrd'))
        sitk.WriteImage(pred_l, os.path.join(pred_dir, cases[i], 'segm_left.nrrd'))


def predict(model_name, onlyEval=False):


    pred_dir = '/home/anneke/projects/uats/code/kits/output/predictions/'

    val_fold = np.load('Folds/val_fold' + str(FOLD_NUM) + '.npy')

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



    # if eval:
    #     val_gt = val_gt.astype(np.uint8)
    #     val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],
    #                    val_gt[:, :, :, :, 4]]
    #     scores = model.evaluate([val_imgs], val_gt_list, batch_size=2, verbose=1)
    #     length = len(model.metrics_names)
    #     for i in range(6, 11):
    #         print("%s: %.16f%%" % (model.metrics_names[i], scores[i]))


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



    # perc = 0.25
    # train(None, None, perc=perc, augmentation=True)
    # #train(None, None, perc=perc, augmentation=False)
    #
    # perc = 0.1
    # train(None, None, perc = perc, augmentation = True)
    # train(None, None, perc=perc, augmentation=False)

    # perc = 1.0
    # train(None, None, perc=perc, augmentation=True)

    #predict(out_dir+'/supervised_F_centered_BB_1_50_0.0005_Perc_0.5_augm.h5', onlyEval=True)

    predict_unlabeled('/home/anneke/projects/uats/code/kits/output/models/supervised_F_centered_BB_1_50_5e-05_Perc_1.0_augm.h5')
