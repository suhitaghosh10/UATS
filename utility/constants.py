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

# prostate
PROSTATE_NR_CLASS = 5
PROSTATE_DIM = [32, 168, 168]
PROSTATE_DATASET = 'prostate'
PROSTATE_LABELLED_TRAIN_NUM = 58
PERCENTAGE_OF_PIXELS = [50, 50, 10, 10, 50]  # pz, tz, us, afs, bg
PROSTATE_VAL_METRIC_KEY_ARR = ['val_pz_dice_coef', 'val_cz_dice_coef', 'val_us_dice_coef', 'val_afs_dice_coef',
                               'val_bg_dice_coef']
