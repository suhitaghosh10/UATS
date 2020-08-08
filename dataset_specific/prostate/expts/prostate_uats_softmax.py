import argparse
import os

import tensorflow as tf

from dataset_specific.prostate.model.uats_scaled import weighted_model
from train.semi_supervised.uats_softmax_A import train
from utility.config import get_metadata
from utility.constants import *
from utility.utils import cleanup

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_num', type=str, default='0', help='GPU Number')
parser.add_argument('-f', '--fold_num', type=int, default=1, help='Fold Number')
parser.add_argument('-e', '--ens_folder_name', type=str, help='ensemble folder name')
parser.add_argument('-d', '--ds', type=str, default=PROSTATE_DATASET_NAME, help='dataset name')





config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

try:
    # fold_num = args.fold_num
    # perc = args.perc
    # temp_path = args.temp_path
    # gpu_num = args.gpu_num
    gpu_num = '0'
    fold_num = 1
    perc = 1.0
    temp_path = 'sadv1'
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    metadata = get_metadata(args.ds, fold_num, perc)
    # Build Model
    wm = weighted_model()
    train(None, None,
          dataset_name= args.ds,
          ens_folder_name=temp_path,
          labelled_perc=perc,
          fold_num=fold_num,
          model_type= wm
              )

finally:
    if os.path.exists(metadata[m_root_temp_path] + temp_path):
        cleanup(metadata[m_root_temp_path] + temp_path)
        print('clean up done!')
