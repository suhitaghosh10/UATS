import argparse
import os

import tensorflow as tf

from dataset_specific.hippocampus.model.entropy import weighted_model
from train.semi_supervised.uats_mc_entropy import train
from utility.config import get_metadata
from utility.constants import *
from utility.utils import cleanup

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_num', type=str, default='0', help='GPU Number')
parser.add_argument('-f', '--fold_num', type=int, default=4, help='Fold Number')
parser.add_argument('-p', '--perc', type=float, default=1.0, help='Percentage of labelled data used') #0.1 0.25 0.5 1.0
parser.add_argument('-t', '--temp_path', type=str, default='temp', help='temporary folder name')
parser.add_argument('-d', '--ds', type=str, default=HIPPOCAMPUS_DATASET_NAME, help='dataset name')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

try:
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    metadata = get_metadata(args.ds)
    # Build Model
    wm = weighted_model()
    train(None, None,
          dataset_name=args.ds,
          ens_folder_name=args.temp_path,
          labelled_perc=args.perc,
          fold_num=args.fold_num,
          model_type=wm
          )

finally:
    if os.path.exists(metadata[m_root_temp_path] + args.temp_path):
        cleanup(metadata[m_root_temp_path] + args.temp_path)
        print('clean up done!')
