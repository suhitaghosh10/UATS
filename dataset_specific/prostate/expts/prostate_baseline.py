import argparse
import os

import tensorflow as tf

from dataset_specific.prostate.model.baseline import weighted_model
from train.supervised.baseline_A import train
from utility.config import get_metadata
from utility.constants import *

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
    gpu_num = '3'
    fold_num = 1
    perc = 0.5
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    metadata = get_metadata(args.ds)
    # Build Model
    wm = weighted_model()
    train(None, None,
          dataset_name=args.ds,
          labelled_perc=perc,
          fold_num=fold_num,
          model_type=wm
          )

finally:
        print('clean up done!')
