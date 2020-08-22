from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping

from old.utils.AugmentationGenerator import *
from utility.config import get_metadata
from utility.constants import *
from utility.parallel_gpu_checkpoint import ModelCheckpointParallel
from utility.utils import get_supervised_val_data, get_supervised_data_generator, makedir


def train(gpu_id, nb_gpus, dataset_name, labelled_perc, fold_num, model_type, is_augmented=True):
    metadata = get_metadata(dataset_name)
    name = 'supervised_F' + str(fold_num) + '_P' + str(labelled_perc)

    data_path = os.path.join(metadata[m_data_path], dataset_name, 'fold_' + str(fold_num) + '_P' + str(labelled_perc), 'train')
    print('data directory:', data_path)
    tb_log_dir = os.path.join(metadata[m_save_path], 'tb', dataset_name, name + '_' + str(metadata[m_lr]) + '/')
    model_name = os.path.join(metadata[m_trained_model_path], dataset_name, name + H5)
    csv_name = os.path.join(metadata[m_save_path], 'csv', dataset_name, name + '.csv')
    dim = metadata[m_dim]
    bs = metadata[m_batch_size]

    num_labeled_train = int(labelled_perc * metadata[m_labelled_train])  # actual labelled data

    print("Labelled Images:", num_labeled_train)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    inp_shape = dim if len(dim)==3 else (dim[0], dim[1], metadata[m_nr_channels])
    model = model_type.build_model(img_shape=inp_shape,
                                   learning_rate=metadata[m_lr],
                                   gpu_id=gpu_id,
                                   nb_gpus=nb_gpus)
    model.summary()

    # callbacks
    print('-' * 30)
    print('Creating callbacks...')
    print('-' * 30)
    makedir(os.path.join(metadata[m_save_path], 'csv', dataset_name))
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
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_EARLY_STOP, min_delta=DELTA)
    cb = [model_checkpoint, tensorboard, csv_logger, es]

    print('Callbacks: ', cb)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    training_generator = get_supervised_data_generator(dataset_name, data_path,
                                                       num_labeled_train,is_augmented)

    steps = (metadata[m_labelled_train] * metadata[m_aug_num]) // bs

    x_val, y_val = get_supervised_val_data(data_path, dim, metadata[m_nr_class], metadata[m_nr_channels])

    history = model.fit_generator(generator=training_generator,
                                  steps_per_epoch=steps,
                                  validation_data=[x_val, y_val],
                                  epochs=NUM_EPOCH,
                                  callbacks=cb
                                  )
    return history
