from utility.constants import *
from utility.config import get_metadata
#from utility.dataset_creation.generate_numpy_supervised import generate_supervised_dataset as gen
from utility.dataset_creation.generate_numpy_uats import generate_uats_dataset as gen
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

ds = PROSTATE_DATASET_NAME
metadata = get_metadata(ds)
gen(ds,
    fold_num=2,
    labelled_perc=0.5,
    ul_imgs_path='/cache/suhita/data/' + ds + '/npy_img_unlabeled.npy',
    folds_root_path='/cache/suhita/data/',
    supervised_model_path=metadata[m_trained_model_path])

# gen(ds,
#     fold_num=3,
#     labelled_perc=0.25,
#     folds_root_path='/cache/suhita/data/')

