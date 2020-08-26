import argparse
import os

import tensorflow as tf

from dataset_specific.skin_2D.model.baseline import weighted_model
from train.supervised.baseline import train
from utility.constants import *

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_num', type=str, default='0', help='GPU Number')
parser.add_argument('-f', '--fold_num', type=int, default=1, help='Fold Number')
parser.add_argument('-p', '--perc', type=float, default=1.0, help='Percentage of Labelled')
parser.add_argument('-e', '--ens_folder_name', type=str, help='ensemble folder name')
parser.add_argument('-d', '--ds', type=str, default=SKIN_DATASET_NAME, help='dataset name')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
args = parser.parse_args()

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    wm = weighted_model()
    train(None, None,
          dataset_name=args.ds,
          labelled_perc=args.perc,
          fold_num=args.fold_num,
          model_type=wm
          )

finally:
        print('clean up done!')
