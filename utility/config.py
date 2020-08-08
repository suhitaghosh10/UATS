from utility.constants import *
import os

#change here
_DATA_PATH = '/cache/suhita/data/prostate/'
_TEMP_PATH = '/data/suhita/temporal/prostate/output/'
_SAVE_PATH = '/data/suhita/experiments/'
_LR = 5e-5


def get_metadata(dataset_name, fold_num, labelled_perc):


    if dataset_name == PROSTATE_DATASET_NAME:
        LABELLED_NUM = 58
        num_train_data = len(os.listdir(os.path.join(_DATA_PATH , 'fold_' + str(fold_num) + '_P' + str(labelled_perc), 'train', IMGS)))
        num_un_labeled_train = num_train_data - LABELLED_NUM

        print("Images Size:", num_train_data)
        print("Unlabeled Size:", num_un_labeled_train)

        return {m_data_path: _DATA_PATH,
                m_save_path: _SAVE_PATH,
                m_root_temp_path: _TEMP_PATH,
                m_batch_size: 2,
                m_lr: _LR,
                m_aug_num: 1,
                m_patients_per_batch: 59,
                m_update_epoch_num: 50,
                m_mc_forward_pass: 10,
                m_nr_class: 5,
                m_nr_channels: 1,
                m_dim: [32, 168, 168],
                m_dataset_name: 'prostate',
                m_labelled_train: LABELLED_NUM,
                m_labelled_val: 20,
                m_unlabelled_train: num_un_labeled_train,
                m_labelled_perc: [50, 50, 10, 10, 50],  # pz, tz, us, afs, bg
                m_metric_keys: ['val_pz_dice_coef', 'val_cz_dice_coef', 'val_us_dice_coef', 'val_afs_dice_coef',
                                'val_bg_dice_coef']

                }
