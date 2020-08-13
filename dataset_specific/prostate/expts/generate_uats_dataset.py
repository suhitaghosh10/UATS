from utility.constants import *
from utility.dataset_creation.generate_numpy_uats import generate_uats_dataset as gen

ds = PROSTATE_DATASET_NAME

gen(ds,
    fold_num=1,
    labelled_perc=0.25,
    ul_imgs_path='/cache/suhita/data/' + ds + '/npy_img_unlabeled.npy',
    folds_root_path='/cache/suhita/data/',
    supervised_model_path='/data/suhita/experiments/prostate/')
