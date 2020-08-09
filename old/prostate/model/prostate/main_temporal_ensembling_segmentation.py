import argparse

import numpy.random as rn
from keras import backend as K
from keras.optimizers import Adam
from sklearn.utils import shuffle

from dataset_specific.prostate.generator import AugmentTypes
from old.preprocess_images import split_supervised_train, make_train_test_dataset, whiten_zca, \
    data_augmentation_tempen
from old.utils.AugmentationGenerator import *
from old.utils.ops import ramp_up_weight, update_unsupervised_target, evaluate, \
    ramp_down_weight, update_weight
from utility.weight_norm import AdamWithWeightnorm


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Ensembling')
    parser.add_argument('--data_path', default='./data/cifar10.npz', type=str, help='path to dataset')
    parser.add_argument('--num_labeled_train', default=4000, type=int,
                        help='the number of labeled data used for supervised training componet')
    parser.add_argument('--num_test', default=10000, type=int,
                        help='the number of data kept out for test')
    parser.add_argument('--num_class', default=10, type=int, help='the number of class')
    parser.add_argument('--num_epoch', default=351, type=int, help='the number of epoch')
    parser.add_argument('--batch_size', default=100, type=int, help='mini batch size')
    parser.add_argument('--ramp_up_period', default=80, type=int, help='ramp-up period of loss function')
    parser.add_argument('--ramp_down_period', default=50, type=int, help='ramp-down period')
    parser.add_argument('--alpha', default=0.6, type=float, help='ensembling momentum')
    parser.add_argument('--weight_max', default=30, type=float, help='related to unsupervised loss component')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate of optimizer')
    parser.add_argument('--whitening_flag', default=True, type=bool, help='Whitening')
    parser.add_argument('--weight_norm_flag', default=True, type=bool,
                        help='Weight normalization is applied. Otherwise Batch normalization is applied')
    parser.add_argument('--augmentation_flag', default=True, type=bool, help='Data augmented')
    parser.add_argument('--trans_range', default=2, type=int, help='random_translation_range')

    args = parser.parse_args()

    return args


def main():
    #50,000 Training -> 4000 supervised data(400 per class) and 46,000 unsupervised data.
    # Prepare args
    #args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    num_labeled_train = 38
    num_test = 20
    ramp_up_period = 100
    ramp_down_period = 50
    num_class = 5
    num_epoch = 400
    batch_size = 2
    weight_max = 20
    learning_rate = 5e-5
    alpha = 0.6
    weight_norm_flag = True
    augmentation_flag = False
    whitening_flag = False
    trans_range = 2

    # Data Preparation
    train_x = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
    train_y = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy')

    test_x = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    test_y = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy')

    # pre-process
    if whitening_flag:
        train_x, test_x = whiten_zca(train_x, test_x)

    if augmentation_flag:
        aug_type = rn.randint(0, 4)
        print('random no ', aug_type)
        augmentation_type = AugmentTypes.FLIP_HORIZ.value
        train_x, train_y = get_image_augmentations_all(augmentation_type, train_x, train_y, 1,
                                                       'final_test_array_imgs.npy', 'final_test_array_gt.npy',
                                                       save_orig=False, save_numpy=False)

    #here we are getting all the data (all have GT). Therefore we split some of them having GT and some not having (unsupervised)
    #ret_dic{labeled_x(4000,32,32,3), labeled_y(4000,),  unlabeled_x(46000,32,32,3)}
    ret_dic = split_supervised_train(train_x, train_y, num_labeled_train)

    ret_dic['val_x'] = test_x
    ret_dic['val_y'] = test_y
    ret_dic = make_train_test_dataset(ret_dic, num_class)

    unsupervised_target = ret_dic['unsupervised_target']
    supervised_label = ret_dic['supervised_label']
    supervised_flag = ret_dic['train_sup_flag']
    unsupervised_weight = ret_dic['unsupervised_weight']

    # make the whole data and labels for training
    y = np.concatenate((unsupervised_target, supervised_label, supervised_flag, unsupervised_weight), axis=-1)

    num_train_data = train_x.shape[0]

    # Build Model
    if weight_norm_flag:
        from old.prostate import build_model
        #from lib.segmentation.weight_norm import AdamWithWeightnorm
        #optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        from old.prostate import build_model
        optimizer = AdamWithWeightnorm(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    else:
        from lib.segmentation.model_BN import build_model
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model = build_model(num_class=num_class)

    model.metrics_tensors += model.outputs
    model.summary()

    # prepare weights and arrays for updates
    gen_weight = ramp_up_weight(ramp_up_period, weight_max * (num_labeled_train / num_train_data))
    gen_lr_weight = ramp_down_weight(ramp_down_period)
    idx_list = [v for v in range(num_train_data)]
    ensemble_prediction = np.zeros((num_train_data, 32, 168, 168, num_class))
    cur_pred = np.zeros((num_train_data, 32, 168, 168, num_class))

    # Training

    for epoch in range(num_epoch):
        print('epoch: ', epoch)
        idx_list = shuffle(idx_list)

        if epoch > num_epoch - ramp_down_period:
            weight_down = next(gen_lr_weight)
            K.set_value(model.optimizer.lr, weight_down * learning_rate)
            K.set_value(model.optimizer.beta_1, 0.4 * weight_down + 0.5)

        ave_loss = 0
        for i in range(0, num_train_data, batch_size):
            target_idx = idx_list[i:i + batch_size]

            if augmentation_flag:
                x1 = data_augmentation_tempen(train_x[target_idx], trans_range)
            else:
                x1 = train_x[target_idx]

            x2 = unsupervised_target[target_idx]
            x3 = supervised_label[target_idx]
            x4 = supervised_flag[target_idx]
            x5 = unsupervised_weight[target_idx]
            # y_t = y[target_idx]
            # mess starts here
            predicted_index = 0
            label_index = 5
            flag_index = 10
            wt_index = 11
            pz = y[target_idx, :, :, :, label_index]
            cz = y[target_idx, :, :, :, label_index + 1]
            us = y[target_idx, :, :, :, label_index + 2]
            afs = y[target_idx, :, :, :, label_index + 3]
            bg = y[target_idx, :, :, :, label_index + 4]

            y_t = [pz, cz, us, afs, bg]
            x_t = [x1, x2, x3, x4, x5]

            loss, pz_loss, cz_loss, us_loss, afs_loss, bg_loss, dice_pz, dice_cz, dice_us, dice_afs, dice_bg, output_0, output_1, output_2, output_3, output_4 = model.train_on_batch(
                x=x_t, y=y_t)

            cur_pred[idx_list[i:i + batch_size], :, :, :, 0] = output_0[:, :, :, :]
            cur_pred[idx_list[i:i + batch_size], :, :, :, 1] = output_1[:, :, :, :]
            cur_pred[idx_list[i:i + batch_size], :, :, :, 2] = output_2[:, :, :, :]
            cur_pred[idx_list[i:i + batch_size], :, :, :, 3] = output_3[:, :, :, :]
            cur_pred[idx_list[i:i + batch_size], :, :, :, 4] = output_4[:, :, :, :]

            ave_loss += loss

        print('Training Loss: ', (ave_loss * batch_size) / (num_train_data), flush=True)

        # Update phase
        next_weight = next(gen_weight)
        y, unsupervised_weight = update_weight(y, unsupervised_weight, next_weight)
        ensemble_prediction, y = update_unsupervised_target(ensemble_prediction, y, num_class, alpha, cur_pred, epoch)

        # Evaluation
        # if epoch % 5 == 0:
        print('Evaluate epoch :  ', epoch, flush=True)
        evaluate(model, num_class, 20, test_x, test_y)

if __name__ == '__main__':
    main()