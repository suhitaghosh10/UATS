import argparse
import os

import tensorflow as tf

from dataset_specific.prostate.model.baseline import weighted_model
from train.supervised.baseline import train
from utility.config import get_metadata
from utility.constants import *

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_num', type=str, default='0', help='GPU Number')
parser.add_argument('-f', '--fold_num', type=int, default=1, help='Fold Number')
parser.add_argument('-p', '--perc', type=float, default=1.0, help='Percentage of labelled data used') #0.1 0.25 0.5 1.0
parser.add_argument('-d', '--ds', type=str, default=PROSTATE_DATASET_NAME, help='dataset name')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

try:
    gpu_num = args.gpu_num
    fold_num = args.fold_num
    perc = args.perc
    
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
