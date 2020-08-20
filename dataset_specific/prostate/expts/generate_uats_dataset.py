from utility.constants import *
from utility.config import get_metadata
from utility.prostate.generate_dataset import generate_uats_dataset, generate_supervised_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

ds = PROSTATE_DATASET_NAME
metadata = get_metadata(ds)
perc=1.0
fold_num=4
generate_supervised_dataset(ds,
    fold_num=fold_num,
    labelled_perc=perc,
    folds_root_path='/cache/suhita/data/',
    seed=1)

generate_uats_dataset(ds,
    fold_num=fold_num,
    labelled_perc=perc,
    ul_imgs_path='/cache/suhita/data/' + ds + '/npy_img_unlabeled.npy',
    folds_root_path='/cache/suhita/data/',
    supervised_model_path=metadata[m_trained_model_path])

# generate_supervised_dataset(ds,
#     fold_num=1,
#     labelled_perc=0.1,
#     folds_root_path='/cache/suhita/data/',
#     seed=1)