import os

import numpy as np

from dataset_specific.prostate.model import weighted_model

learning_rate = 5e-5
MODEL_NAME = '/data/suhita/temporal/supervised_F1_6.h5'

IMG = '/cache/suhita/data/final_test_array_imgs.npy'
GT = '/cache/suhita/data/final_test_array_GT.npy'
# IMG = '/cache/suhita/data/npy_img_unlabeled.npy'
# GT = ''
NUM_CLASS = 5
num_epoch = 351
batch_size = 2


def predict(model_name, eval=True, out_npy=None, gpu_id=None, nb_gpus=None):
    out_dir = './'
    val_imgs = np.load(IMG)


    wm = weighted_model()
    model = wm.build_model(learning_rate=learning_rate, gpu_id=gpu_id,
                           nb_gpus=nb_gpus, trained_model=model_name)
    print('load_weights')
    out = model.predict([val_imgs], batch_size=1, verbose=1)
    np.save(os.path.join(out_npy), out)
    if eval:
        val_gt = np.load(GT)
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
    gpu_id = '3'
    # gpu_id = '0'
    # gpu = "GPU:0"  # gpu_id (default id is first of listed in parameters)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    ##config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # set_session(tf.Session(config=config))

    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    predict(MODEL_NAME, eval=True, out_npy='/data/suhita/temporal/p_30.npy', gpu_id=None, nb_gpus=None)
