import os

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from utility.callbacks.temporal import TemporalCallback
from utility.config import get_metadata
from utility.constants import *
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel
from utility.utils import get_temporal_val_data, get_temporal_data_generator


def train(gpu_id, nb_gpus, dataset_name, ens_folder_name, labelled_perc, fold_num, model_type, is_augmented=True):
    global metadata
    metadata = get_metadata(dataset_name, fold_num, labelled_perc)
    name = 'temporal_F' + str(fold_num) + '_Perct_Labelled_' + str(labelled_perc)

    data_path = os.path.join(metadata[m_data_path], 'fold_' + str(fold_num) + '_P' + str(labelled_perc), 'train')
    print('data directory:', data_path)
    tb_log_dir = os.path.join(metadata[m_save_path], 'tb', dataset_name, name + '_' + str(metadata[m_lr]) + '/')
    model_name = os.path.join(metadata[m_save_path], 'model', 'temporal', dataset_name, name + H5)
    csv_name = os.path.join(metadata[m_save_path], 'csv', dataset_name, name + '.csv')
    ens_path = os.path.join(metadata[m_root_temp_path], ens_folder_name)
    dim = metadata[m_dim]
    bs = metadata[m_batch_size]
    num_train_data = metadata[m_labelled_train] + metadata[m_unlabelled_train]

    num_labeled_train = int(labelled_perc * metadata[m_labelled_train])

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = model_type.build_model(img_shape=(dim[0], dim[1], dim[2]),
                                   learning_rate=metadata[m_lr],
                                   gpu_id=gpu_id,
                                   nb_gpus=nb_gpus)
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
                           metadata[m_patients_per_batch], metadata[m_nr_class], bs, dataset_name)
    lcb = model_type.LossCallback()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_EARLY_STOP, min_delta=DELTA)
    cb = [model_checkpoint, tcb, tensorboard, lcb, csv_logger, es]

    print('Callbacks: ', cb)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    training_generator = get_temporal_data_generator(dataset_name, data_path, ens_path, num_train_data,
                                                     num_labeled_train, bs,
                                                     is_augmented)
    steps = (metadata[m_labelled_train] * metadata[m_aug_num]) // bs
    # steps=2

    x_val, y_val = get_temporal_val_data(data_path, dim, metadata[m_nr_class], metadata[m_nr_channels])

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=NUM_EPOCH,
                                  callbacks=cb
                                  )
    return history
