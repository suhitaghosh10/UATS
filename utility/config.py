from utility.constants import *

# change here
_DATA_PATH = '/cache/suhita/data/'
_TEMP_PATH = '/data/suhita/temporal/output/'
_SAVE_PATH ='/data/suhita/experiments/'
_TRAINED_MODEL_PATH = '/data/suhita/experiments/model/supervised/'


def get_metadata(dataset_name):
    if dataset_name == PROSTATE_DATASET_NAME:
        return {m_dataset_name: PROSTATE_DATASET_NAME,
                # hyper-param
                m_data_path: _DATA_PATH,
                m_save_path: _SAVE_PATH ,
                m_root_temp_path: _TEMP_PATH,
                m_trained_model_path: _TRAINED_MODEL_PATH,
                m_lr: 5e-5,
                m_batch_size: 2,
                m_aug_num: 1,
                m_patients_per_batch: 59,
                m_update_epoch_num: 50,
                m_mc_forward_pass: 10,
                m_labelled_perc: [25, 50, 25, 25, 25],  # pz, tz, us, afs, bg #for 0.1, 0.25, 0.5
#_PERC = [50, 50, 10, 10, 50] # pz, tz, us, afs, bg #for only 1.0 experiments,

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

    elif dataset_name == SKIN_DATASET_NAME:
                    return {m_dataset_name: SKIN_DATASET_NAME,
                            # hyper-param
                            m_raw_data_path:'/cache/suhita/skin/preprocessed',
                            m_folds_path:'/cache/suhita/skin/Folds',
                            m_data_path: _DATA_PATH,
                            m_save_path: _SAVE_PATH,
                            m_root_temp_path: _TEMP_PATH,
                            m_trained_model_path: _TRAINED_MODEL_PATH,
                            m_lr: 1e-7,
                            m_batch_size: 8,
                            m_aug_num: 5,
                            m_patients_per_batch: 250,
                            m_update_epoch_num: 50,
                            m_mc_forward_pass: 10,
                            m_labelled_perc: [50, 50],

                            # param
                            m_nr_class: 2,
                            m_nr_channels: 3,
                            m_dim: [192, 256],
                            m_labelled_train: 1570,
                            m_labelled_val: 523,
                            m_unlabelled_train: 1000,
                            m_metric_keys: ['val_skin_dice_coef', 'val_bg_dice_coef']

                            }
    elif dataset_name == HIPPOCAMPUS_DATASET_NAME:
                    return {m_dataset_name: HIPPOCAMPUS_DATASET_NAME,
                            # hyper-param
                            m_raw_data_path:'/cache/suhita/hippo/preprocessed',
                            m_folds_path:'/cache/suhita/hippo/Folds',
                            m_data_path: _DATA_PATH,
                            m_save_path: _SAVE_PATH,
                            m_root_temp_path: _TEMP_PATH,
                            m_trained_model_path: _TRAINED_MODEL_PATH,
                            m_lr: 4e-5,
                            m_batch_size: 4,
                            m_aug_num: 5,
                            m_patients_per_batch: num_un_labeled_train // 3,
                            m_update_epoch_num: 50,
                            m_mc_forward_pass: 10,
                            m_labelled_perc: [50, 50, 50],

                            # param
                            m_nr_class: 3,
                            m_nr_channels: 1,
                            m_dim: [48, 64, 48],
                            m_labelled_train: TODO,
                            m_labelled_val: TODO,
                            m_unlabelled_train: TODO,
                            m_metric_keys: ['val_bg_dice_coef', 'val_z1_dice_coef', 'val_z2_dice_coef']

                            }

