import sys
sys.path.append('../')

from kits import preprocess

import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from skin_2D.data_generation import DataGenerator as train_gen
from skin_2D.model import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel
from lib.segmentation.utils import get_complete_array
from kits import utils
import SimpleITK as sitk

data_path = '/cache/anneke/skin/preprocessed/labelled/train'
out_dir = '/home/anneke/projects/uats/code/skin_2D/output/models/'


learning_rate = 5e-5
AUGMENTATION_NO = 5
TRAIN_NUM = 1000
#PERC = 0.25
FOLD_NUM = 1


NUM_CLASS = 1
num_epoch = 1000
batch_size = 8
DIM = [192,256]
N_CHANNELS = 3


def remove_tumor_segmentation(arr):
    arr[arr > 1] = 1
    return arr


def train(gpu_id, nb_gpus, trained_model=None, perc=1.0, augmentation = False):

    if augmentation:
        augm = '_augm'
    else:
        augm = ''

    NAME = 'supervised_sfs32_F_' + str(FOLD_NUM) + '_' + str(TRAIN_NUM) + '_' + str(
        learning_rate) + '_Perc_' + str(perc)+augm
    CSV_NAME = out_dir + NAME + '.csv'

    TB_LOG_DIR = out_dir + NAME + str(learning_rate) + '/'
    MODEL_NAME = out_dir + NAME + '.h5'
    TRAINED_MODEL_PATH = MODEL_NAME

    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0],DIM[1],N_CHANNELS), learning_rate=learning_rate)

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

    np.random.seed(1234)
    np.random.shuffle(train_fold)
    print(train_fold[0:10])

    train_fold = train_fold[:int(nr_samples*perc)]

    train_id_list = []
    for i in range(train_fold.shape[0]):
        train_id_list.append(os.path.join(train_fold[i]))


    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag

    cb = [model_checkpoint, tensorboard, es, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (DIM[0],DIM[1]),
              'batch_size': batch_size,
              'n_channels': 3}

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

    val_img_arr = np.zeros((val_fold.shape[0], DIM[0],DIM[1],N_CHANNELS), dtype = float)

    val_GT_arr = np.zeros((val_fold.shape[0], DIM[0],DIM[1],1), dtype=int)


    for i in range(val_fold.shape[0]):
        val_img_arr[i] = np.load(os.path.join(data_path, 'imgs', val_fold[i]))
        val_GT_arr[i, :, :, 0] = np.load(os.path.join(data_path, 'GT', val_fold[i][:-4]+'_segmentation.npy', ))[:,:,0]

    val_img_arr = val_img_arr / 255
    val_GT_arr = val_GT_arr / 255

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


    # DIM = [80,200,200]
    # SPACING = [4.0, 1.0, 1.0]


    img_dir = '/data/anneke/kits_challenge/kits19/data/preprocessed_unlabeled'
    cases = sorted(os.listdir(img_dir))

    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0], DIM[1], N_CHANNELS), learning_rate=learning_rate)
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


def predict(model_name, pred_dir, onlyEval=False):

    import cv2

    input_dir = '/cache/anneke/skin/preprocessed/labelled/test'


    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    test_imgs = os.listdir(input_dir+'/imgs')
    val_img_arr = np.zeros((len(test_imgs), DIM[0], DIM[1], N_CHANNELS), dtype=float)

    val_GT_arr = np.zeros(((len(test_imgs)), DIM[0], DIM[1], 1), dtype=int)

    for i in range(len(test_imgs)):
        val_img_arr[i] = np.load(os.path.join(input_dir, 'imgs',test_imgs[i]))
        val_GT_arr[i, :, :, 0] = np.load(os.path.join(input_dir, 'GT', test_imgs[i][:-4] + '_segmentation.npy', ))[:, :,
                                 0]

    val_img_arr = val_img_arr / 255
    val_GT_arr = val_GT_arr / 255


    print('load_weights')
    wm = weighted_model()
    model = wm.build_model(img_shape=(DIM[0],DIM[1], N_CHANNELS), learning_rate=learning_rate)
    model.load_weights(model_name)

    if onlyEval:
        out_value = model.evaluate(val_img_arr, val_GT_arr, batch_size=1, verbose=0)
        print(out_value)
    else:
        out = model.predict(val_img_arr, batch_size=2, verbose=1)
        #np.save(os.path.join(out_dir, 'predicted.npy'), out)
        for i in range(out.shape[0]):
            img = np.load(os.path.join(input_dir, 'imgs', test_imgs[i]))
            cv2.imwrite(os.path.join(pred_dir, test_imgs[i][:-4]+'.jpg'), img)
            segm = out[i,:,:,0].astype(np.uint8)
            segm_3ch = np.zeros([segm.shape[0], segm.shape[1], 3], dtype=np.uint8)
            segm_3ch[:,:,0] = segm*255
            segm_3ch[:, :, 1] = segm * 255
            segm_3ch[:, :, 2] = segm * 255
            filename = os.path.join(pred_dir, test_imgs[i][:-4]+'_segm.jpg')
            cv2.imwrite(filename, segm_3ch)




    # if eval:
    #     val_gt = val_gt.astype(np.uint8)
    #     val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],
    #                    val_gt[:, :, :, :, 4]]
    #     scores = model.evaluate([val_imgs], val_gt_list, batch_size=2, verbose=1)
    #     length = len(model.metrics_names)
    #     for i in range(6, 11):
    #         print("%s: %.16f%%" % (model.metrics_names[i], scores[i]))


if __name__ == '__main__':

    GPU_ID = '0'
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


    #
    # perc = 0.25
    # train(None, None, perc=perc, augmentation=True)
    # perc = 0.1
    # train(None, None, perc = perc, augmentation = True)
    # perc = 0.05
    # train(None, None, perc=perc, augmentation=True)

    # perc = 1.0
    # train(None, None, perc=perc, augmentation=True)

    perc = 1.0
    model_name = '/home/anneke/projects/uats/code/skin_2D/output/models/supervised_sfs32_F_1_1000_5e-05_Perc_'+str(perc)+'_augm.h5'
    pred_dir = '/home/anneke/projects/uats/code/skin_2D/output/predictions/'+str(perc)
    predict(model_name= model_name, pred_dir = pred_dir, onlyEval=False)
    #
    # predict_unlabeled('/home/anneke/projects/uats/code/kits/output/models/supervised_F_centered_BB_1_50_5e-05_Perc_1.0_augm.h5')
