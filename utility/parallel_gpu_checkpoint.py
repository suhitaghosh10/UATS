import warnings

import keras
import numpy as np


class ModelCheckpointParallel(keras.callbacks.Callback):
    """

    borrow from: https://github.com/rmkemker/main/blob/master/machine_learning/model_checkpoint_parallel.py

    Save the train after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the train checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the train file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best train according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the train's weights will be
            saved (`train.save_weights(filepath)`), else the full train
            is saved (`train.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 at_batch_end=None,
                 at_epoch_end=True,
                 mode='auto', period=1):
        super(ModelCheckpointParallel, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpointParallel mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            print("Saving train at batch end!")
            self.on_model_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_save(epoch, 0, logs=logs)
        self.current_epoch = epoch + 1

    def on_model_save(self, epoch, iter, logs=None):
        """ save the train to hdf5. Code mostly from keras core """

        logs = logs or {}
        num_outputs = len(self.model.outputs)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, iter=iter, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best train only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: Iter%05d: %s improved from %0.5f to %0.5f,'
                                  ' saving train to %s'
                                  % (epoch, iter, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.layers[-(num_outputs + 1)].save_weights(filepath, overwrite=True)
                        else:
                            self.model.layers[-(num_outputs + 1)].save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d Iter%05d: %s did not improve' %
                                  (epoch, iter, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving train to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.layers[-(num_outputs + 1)].save_weights(filepath, overwrite=True)
                else:
                    self.model.layers[-(num_outputs + 1)].save(filepath, overwrite=True)
