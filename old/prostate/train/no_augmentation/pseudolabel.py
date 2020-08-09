import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger

from dataset_specific.prostate.generator.baseline import DataGenerator as gen
from dataset_specific.prostate.model import weighted_model
from old.utils.AugmentationGenerator import *
from old.utils.ops import ramp_down_weight
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel

# 294 Training 58 have gt
learning_rate = 5e-5
FOLD_NUM = 1
TB_LOG_DIR = '/data/suhita/temporal/tb/variance_mcdropout/pseudo_F' + str(FOLD_NUM) + '_' + str(
    learning_rate) + '/'
MODEL_NAME = '/data/suhita/temporal/pseudo_F' + str(learning_rate)

CSV_NAME = '/data/suhita/temporal/CSV/Pseudolabel_F' + str(FOLD_NUM) + '.csv'

TRAIN_IMGS_PATH = '/cache/suhita/data/training/imgs/'
TRAIN_GT_PATH = '/cache/suhita/data/training/gt/'

TEST_IMGS_PATH = '/cache/suhita/data/test_anneke/imgs/'
TEST_GT_PATH = '/cache/suhita/data/test_anneke/gt/'

TRAINED_MODEL_PATH = '/data/suhita/data/train.h5'

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
    gen_lr_weight = ramp_down_weight(ramp_down_period)

    print('-' * 30)
    print('Creating and compiling train...')
    print('-' * 30)

    # train.metrics_tensors += train.outputs
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
    LRDecay = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min', min_lr=1e-8,
                                epsilon=0.01)

    # datagen listmake_dataset
    train_id_list = [str(i) for i in np.arange(294)]
    cb = [model_checkpoint, tensorboard, LRDecay, csv_logger]

    print('BATCH Size = ', batch_size)

    print('Callbacks: ', cb)
    params = {'dim': (32, 168, 168),
              'batch_size': batch_size}

    print('-' * 30)
    print('Fitting train...')
    print('-' * 30)
    training_generator = gen(TRAIN_IMGS_PATH,
                             TRAIN_GT_PATH,
                             train_id_list,
                             **params)

    steps = 294 / batch_size
    # steps=2

    # val_imgs = get_complete_array(TEST_IMGS_PATH)
    # val_gt = get_complete_array(TEST_GT_PATH, dtype='int8')
    # val_mask = np.zeros_like(val_gt)
    # val_gt = val_gt.astype(np.uint8)
    # val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],val_gt[:, :, :, :, 4]]
    val_id_list = [str(i) for i in np.arange(20)]
    val_generator = gen(TEST_IMGS_PATH,
                        TEST_GT_PATH,
                        val_id_list,
                        **params)

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=val_generator,
                                  epochs=num_epoch,
                                  callbacks=cb
                                  )

    # workers=4)
    # train.save('temporal_max_ramp_final.h5')


def predict(val_imgs, val_gt, model_name, evaluate=True):
    nrChanels = 1
    out_dir = './'
    val_imgs = np.load('/cache/suhita/data/test_anneke/final_test_array_imgs.npy')
    val_gt = np.load('/cache/suhita/data/test_anneke/final_test_array_GT.npy')

    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=None,
                           nb_gpus=None,
                           trained_model=model_name)
    print('load_weights')
    # train.load_weights(TRAINED_MODEL_PATH)
    out = model.predict([val_imgs], batch_size=1, verbose=1)
    np.save(os.path.join(out_dir, 'original_predicted.npy'), out)

    if evaluate:
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
    gpu_id = '0'
    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train(None, None, trained_model=TRAINED_MODEL_PATH)
    # train(gpu, nb_gpus, trained_model=TRAINED_MODEL_PATH)
    # val_x = np.load('/cache/suhita/data/validation/valArray_imgs_fold1.npy')
    # val_y = np.load('/cache/suhita/data/validation/valArray_GT_fold1.npy').astype('int8')

    val_x = TEST_IMGS_PATH
    val_y = TEST_GT_PATH
    predict(val_x, val_y, MODEL_NAME, evaluate=True)