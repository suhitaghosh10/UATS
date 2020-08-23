# do not change

NPY = '.npy'
VAL_IMGS_PATH = '/val/imgs/'
VAL_GT_PATH = '/val/gt/'
ALPHA = 0.6
H5 = '.h5'
ENS_GT = 'ens_gt'
FLAG = 'flag'
IMGS = 'imgs'
GT = 'gt'
PATIENCE_EARLY_STOP = 30
DELTA = 0.0005
NUM_EPOCH = 1000

# dataset names
PROSTATE_DATASET_NAME = 'prostate'
SKIN_DATASET_NAME = 'skin'

# model types
MODEL_SSL_UATS_SM = 'uats_softmax'
MODEL_SSL_UATS_ENTROPY = 'uats_mc_entropy'
MODEL_SSL_UATS_VARIANCE = 'uats_mc_variance'
MODEL_SSL_TEMPORAL = 'uats_softmax'
MODEL_SSL_BAI = 'bai'
MODEL_SSL_PSEUDO_SB = 'pseudo_save_best'
MODEL_SL_BASELINE = 'baseline'
MODEL_SL_TEMPORAL = 'supervised_temporal'

# metadata
m_raw_data_path = 'raw_data_path'
m_data_path = 'data_path'
m_folds_path = 'folds_numpy_path'
m_save_path = 'save_path'
m_root_temp_path = 'temp_path'
m_trained_model_path = 'trained_model'
m_batch_size = 'batch_size'
m_nr_channels = 'nr_channels'
m_lr = 'lr'
m_aug_num = 'aug_num'
m_patients_per_batch = 'patients_per_batch'
m_update_epoch_num = 'update_epoch_num'
m_mc_forward_pass = 'mc_forward_pass'
m_nr_class = 'nr_class'
m_dim = 'dim'
m_dataset_name = 'dataset'
m_labelled_train = 'train_lablelled_num'
m_labelled_val = 'val_num'
m_unlabelled_train = 'train_unlabelled_num'
m_labelled_perc = 'labelled_perc'
m_metric_keys = 'metric_keys'
