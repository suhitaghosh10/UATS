from utility.constants import *

# change here
_DATA_PATH = '/cache/suhita/data/prostate/'
_TEMP_PATH = '/data/suhita/temporal/prostate/output/'
_SAVE_PATH = '/data/suhita/experiments/'
_TRAINED_MODEL_PATH = '/data/suhita/experiments/model/supervised/'
_UNLABELED_IMG_NUMPY = 'npy_img_unlabeled.npy'
_LR = 5e-5


def get_metadata(dataset_name):
    if dataset_name == PROSTATE_DATASET_NAME:
        return {m_dataset_name: 'prostate',
                # hyper-param
                m_data_path: _DATA_PATH,
                m_save_path: _SAVE_PATH,
                m_root_temp_path: _TEMP_PATH,
                m_trained_model_path: _TRAINED_MODEL_PATH,
                m_lr: _LR,
                m_batch_size: 2,
                m_aug_num: 1,
                m_patients_per_batch: 59,
                m_update_epoch_num: 50,
                m_mc_forward_pass: 10,
                m_labelled_perc: [50, 50, 10, 10, 50],  # pz, tz, us, afs, bg

                # param
                m_nr_class: 5,
                m_nr_channels: 1,
                m_dim: [32, 168, 168],
                m_labelled_train: 58,
                m_labelled_val: 20,
                m_unlabelled_train: 236,
                m_metric_keys: ['val_pz_dice_coef', 'val_cz_dice_coef', 'val_us_dice_coef', 'val_afs_dice_coef',
                                'val_bg_dice_coef']

                }
