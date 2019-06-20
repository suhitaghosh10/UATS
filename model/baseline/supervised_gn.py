import os

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from generator.train_data_gen_baseline import DataGenerator as val_gen
from generator.val_data_gen_baseline import DataGenerator as train_gen
from lib.segmentation.model_GN import weighted_model
from lib.segmentation.parallel_gpu_checkpoint import ModelCheckpointParallel

learning_rate = 5e-5
TB_LOG_DIR = '/home/suhita/zonals/temporal/tb/variance_mcdropout/supervised_gn' + str(learning_rate) + '/'
MODEL_NAME = '/home/suhita/zonals/temporal/supervised_gn.h5'

TRAIN_IMGS_PATH = '/home/suhita/zonals/data/training/supervised/imgs/'
TRAIN_GT_PATH = '/home/suhita/zonals/data/training/supervised/gt/'

VAL_IMGS_PATH = '/home/suhita/zonals/data/validation/imgs/'
VAL_GT_PATH = '/home/suhita/zonals/data/validation/gt/'

TEST_IMGS_PATH = '/home/suhita/zonals/data/test_anneke/imgs/'
TEST_GT_PATH = '/home/suhita/zonals/data/test_anneke/gt/'

TRAINED_MODEL_PATH = MODEL_NAME

NUM_CLASS = 5
num_epoch = 351
batch_size = 2
IMGS_PER_ENS_BATCH = 100

# hyper-params
# UPDATE_WTS_AFTER_EPOCH = 1
ENSEMBLE_NO = 1
SAVE_WTS_AFTR_EPOCH = 0
ramp_up_period = 50
ramp_down_period = 50
# weight_max = 40
weight_max = 30

alpha = 0.6
VAR_THRESHOLD = 0.5


def train(gpu_id, nb_gpus, trained_model=None):
    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=trained_model)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model.summary()

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
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)

    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(77)]

    # del unsupervised_target, unsupervised_weight, supervised_flag, imgs
    # del supervised_flag

    cb = [model_checkpoint, tensorboard, LRDecay]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    training_generator = train_gen(TRAIN_IMGS_PATH,
                                   TRAIN_GT_PATH,
                                   train_id_list,
                                   **params)

    steps = 78 / batch_size
    # steps=2

    # val_imgs = get_complete_array(TEST_IMGS_PATH)
    # val_gt = get_complete_array(TEST_GT_PATH, dtype='int8')
    # val_mask = np.zeros_like(val_gt)
    # val_gt = val_gt.astype(np.uint8)
    # val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],val_gt[:, :, :, :, 4]]
    val_id_list = [str(i) for i in np.arange(20)]
    val_generator = val_gen(VAL_IMGS_PATH,
                            VAL_GT_PATH,
                            val_id_list,
                            **params)

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=val_generator,
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    # workers=4)
    # model.save('temporal_max_ramp_final.h5')


def predict(model_name, eval=True):
    out_dir = './'
    val_imgs = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_imgs.npy')
    val_gt = np.load('/home/suhita/zonals/data/test_anneke/final_test_array_GT.npy')

    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None, trained_model=model_name)
    print('load_weights')
    # model.load_weights(TRAINED_MODEL_PATH)
    out = model.predict([val_imgs], batch_size=1, verbose=1)
    np.save(os.path.join(out_dir, 'gn_predicted.npy'), out)
    if eval:
        val_gt = val_gt.astype(np.uint8)
        val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],
                       val_gt[:, :, :, :, 4]]
        scores = model.evaluate([val_imgs], val_gt_list, batch_size=2, verbose=1)
        length = len(model.metrics_names)
        for i in range(6, 11):
            print("%s: %.16f%%" % (model.metrics_names[i], scores[i]))


if __name__ == '__main__':
    gpu = '/GPU:0'
    # gpu = '/GPU:0'
    batch_size = 2
    gpu_id = '2, 3'
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

    train(gpu, nb_gpus, trained_model=None)
    # train(gpu, nb_gpus, trained_model=TRAINED_MODEL_PATH)
    # val_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = TEST_IMGS_PATH
    val_y = TEST_GT_PATH
    # predict(TRAINED_MODEL_PATH, eval=False)
