import os
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from utility.callbacks.uats_softmax import TemporalCallback
from utility.config import get_metadata
from utility.constants import *
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel
from utility.utils import get_uats_val_data, get_uats_data_generator, makedir
import tensorflow as tf


def train(gpu_id, nb_gpus, dataset_name, ens_folder_name, labelled_perc, fold_num, model_type, is_augmented=True, early_stop=True):
    metadata = get_metadata(dataset_name)
    name = 'uats_softmax_F' + str(fold_num) + '_Perct_Labelled_' + str(labelled_perc)

    data_path = os.path.join(metadata[m_data_path], dataset_name, 'fold_' + str(fold_num) + '_P' + str(labelled_perc), 'train')
    print('data directory:', data_path)
    tb_log_dir = os.path.join(metadata[m_save_path], 'tb', dataset_name, name + '_' + str(metadata[m_lr]) + '/')
    model_name = os.path.join(metadata[m_save_path], 'model', 'uats', dataset_name, name + H5)
    makedir(os.path.join(metadata[m_save_path], 'model', 'uats', dataset_name))

    csv_name = os.path.join(metadata[m_save_path], 'csv', dataset_name, name + '.csv')
    makedir(os.path.join(metadata[m_save_path], 'csv', dataset_name))

    ens_path = os.path.join(metadata[m_root_temp_path], ens_folder_name)
    trained_model_path = os.path.join(metadata[m_trained_model_path], dataset_name, 'supervised_F' + str(fold_num) + '_P' + str(
        labelled_perc) + H5)
    dim = metadata[m_dim]
    inp_shape = dim if len(dim)==3 else [dim[0], dim[1], metadata[m_nr_channels]]
    bs = metadata[m_batch_size]

    num_labeled_train = int(labelled_perc * metadata[m_labelled_train])  # actual labelled data
    num_ul = metadata[m_unlabelled_train]
    num_train_data = num_labeled_train + num_ul

    print("Labelled Images:", num_labeled_train)
    print("Unlabeled Images:", metadata[m_unlabelled_train])
    print("Total Images:", num_train_data)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = model_type.build_model(img_shape=inp_shape,
                                   learning_rate=metadata[m_lr],
                                   gpu_id=gpu_id,
                                   nb_gpus=nb_gpus,
                                   trained_model=trained_model_path,
                                   temp=1)
    model.summary()

    # callbacks
    print('-' * 30)
    print('Creating callbacks...')
    print('-' * 30)
    csv_logger = CSVLogger(csv_name, append=True, separator=';')
    if nb_gpus is not None and nb_gpus > 1:
        model_checkpoint = ModelCheckpointParallel(model_name,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   verbose=1,
                                                   mode='min')
    else:
        model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1,
                                           mode='min')

    tensorboard = TensorBoard(log_dir=tb_log_dir, write_graph=False, write_grads=False, histogram_freq=0,
                              batch_size=1, write_images=False)

    tcb = TemporalCallback(dim, data_path, ens_path, metadata[m_save_path], num_train_data, num_labeled_train,
                           metadata[m_patients_per_batch], metadata[m_labelled_perc], metadata[m_metric_keys],
                           metadata[m_nr_class], bs, dataset_name)

    lcb = model_type.LossCallback()
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger]
    if early_stop:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_EARLY_STOP, min_delta=DELTA)
        cb.append(es)

    print('Callbacks: ', cb)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    training_generator = get_uats_data_generator(dataset_name, data_path, ens_path, num_train_data, num_labeled_train,
                                                 bs,
                                                 is_augmented)

    steps = ((metadata[m_labelled_train] + num_ul) * metadata[m_aug_num]) // bs

    x_val, y_val = get_uats_val_data(data_path, metadata[m_dim], metadata[m_nr_class], metadata[m_nr_channels])

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=NUM_EPOCH,
                                  callbacks=cb
                                  )
    return history
