import os

import numpy as np
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import concatenate, Input, Conv3D, MaxPooling3D, Conv3DTranspose, Lambda, \
    BatchNormalization, Dropout
from keras.callbacks import CSVLogger

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_generation.generator.val_data_gen import ValDataGenerator
from data_generation.generator.train_data_gen import DataGenerator

LOG_DIR = './tensorboard_logs/'

class weighted_model:


    def get_Tversky(alpha=.3, beta=.7, verb=0):
        def Tversky(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            G_P = alpha * K.sum((1 - y_true_f) * y_pred_f)  # G not P
            P_G = beta * K.sum(y_true_f * (1 - y_pred_f))  # P not G
            return (intersection + smooth) / (intersection + smooth + G_P + P_G)

        def Tversky_loss(y_true, y_pred):
            return -Tversky(y_true, y_pred)

        return Tversky, Tversky_loss

    # Tversky, Tversky_loss = get_Tversky(alpha = .3,beta= .7,verb=0)
    # Metrics = [dice_coef_loss ,Tversky

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def jaccard_distance(self, y_true, y_pred):
        smooth = 100
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

    def jaccard_distance(self, y_true, y_pred):
        return -self.jaccard_distance(y_true, y_pred)


    def dice_coef_loss(self, y_true, y_pred):
        dice_loss = -self.dice_coef(y_true, y_pred)

        return dice_loss

    def complex_loss(self, mask):
        """custom loss function"""
        print(mask.shape)
        epsilon = 1e-08

        def loss(y_true, y_pred, axis=(-4, -3, -2)):
            print(y_true.shape)
            # y_true = y_true[:,:,:,0:2]

            dice_numerator = 2.0 * K.sum(y_pred * y_true * mask, axis=axis) + epsilon / 2
            dice_denominator = K.sum(y_true * mask, axis=axis) + \
                               K.sum(y_pred * mask, axis=axis) + epsilon
            w_dice_coef = (dice_numerator / dice_denominator)
            #ce = K.categorical_crossentropy(y_pred, y_pred)

            #loss_c = np.multiply(w_dice_coef, ce)
            return -K.mean(w_dice_coef)

        return loss

    def weighted_dice_coef_loss(self, y_true, y_pred):
        return -self.afs_weight*self.dice_coef(y_true, y_pred)



    # downsampling, analysis path
    def downLayer(self, inputLayer, filterSize, i, bn=False):

        conv = Conv3D(filterSize, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(inputLayer)
        if bn:
            conv = BatchNormalization()(conv)
        conv = Conv3D(filterSize * 2, (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling3D(pool_size=(1, 2, 2))(conv)

        return pool, conv


    # upsampling, synthesis path
    def upLayer(self, inputLayer, concatLayer, filterSize, i, bn=False, do= False):

        up = Conv3DTranspose(filterSize, (2, 2, 2), strides=(1, 2, 2), activation='relu', padding='same',  name='up'+str(i))(inputLayer)
       # print( concatLayer.shape)
        up = concatenate([up, concatLayer])
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_1')(up)
        if bn:
            conv = BatchNormalization()(conv)
        if do:
            conv = Dropout(0.5, seed = 3, name='Dropout_' + str(i))(conv)
        conv = Conv3D(int(filterSize/2), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(i)+'_2')(conv)
        if bn:
            conv = BatchNormalization()(conv)

        return conv



    def get_net(self, nrInputChannels, learningRate=5e-5, bn = True, do = False, mask = False):

        sfs = 16 # start filter size

        if mask:
            inputs = Input((32, 168, 168, nrInputChannels))
        else:
            inputs = Input((32, 168, 168, nrInputChannels))
        yTrueInputs = Input((32, 168, 168, 5))

        conv1, conv1_b_m = self.downLayer(inputs, sfs, 1, bn)
        conv2, conv2_b_m = self.downLayer(conv1, sfs*2, 2, bn)

        conv3 = Conv3D(sfs*4, (3, 3, 3), activation='relu', padding='same', name='conv' + str(3) + '_1')(conv2)
        if bn:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(sfs * 8, (3, 3, 3), activation='relu', padding='same',  name='conv' + str(3) + '_2')(conv3)
        if bn:
            conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        #conv3, conv3_b_m = downLayer(conv2, sfs*4, 3, bn)

        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_1')(pool3)
        if bn:
            conv4 = BatchNormalization()(conv4)
        if do:
            conv4= Dropout(0.5, seed = 4, name='Dropout_' + str(4))(conv4)
        conv4 = Conv3D(sfs*16 , (3, 3, 3), activation='relu', padding='same',  name='conv4_2')(conv4)
        if bn:
            conv4 = BatchNormalization()(conv4)

        #conv5 = upLayer(conv4, conv3_b_m, sfs*16, 5, bn, do)
        up1 = Conv3DTranspose(sfs*16, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same', name='up'+str(5))(conv4)
        up1 = concatenate([up1, conv3])
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same',  name='conv'+str(5)+'_1')(up1)
        if bn:
            conv5 = BatchNormalization()(conv5)
        if do:
            conv5 = Dropout(0.5, seed = 5, name='Dropout_' + str(5))(conv5)
        conv5 = Conv3D(int(sfs*8), (3, 3, 3), activation='relu', padding='same', name='conv'+str(5)+'_2')(conv5)
        if bn:
            conv5 = BatchNormalization()(conv5)

        conv6 = self.upLayer(conv5, conv2_b_m, sfs*8, 6, bn, do)
        conv7 = self.upLayer(conv6, conv1_b_m, sfs*4, 7, bn, do)



        conv_out = Conv3D(5, (1, 1, 1), activation='softmax', name='conv_final_softmax')(conv7)

        pz = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(conv_out)
        cz = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(conv_out)
        us = Lambda(lambda x: x[:, :, :, :, 2], name='us')(conv_out)
        afs = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(conv_out)
        bg = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(conv_out)

        pzw = Lambda(lambda x: x[:, :, :, :, 0], name='pz')(yTrueInputs)
        czw = Lambda(lambda x: x[:, :, :, :, 1], name='cz')(yTrueInputs)
        usw = Lambda(lambda x: x[:, :, :, :, 2], name='us')(yTrueInputs)
        afsw = Lambda(lambda x: x[:, :, :, :, 3], name='afs')(yTrueInputs)
        bgw = Lambda(lambda x: x[:, :, :, :, 4], name='bg')(yTrueInputs)

        model = Model(inputs=[inputs, yTrueInputs], outputs=[pz, cz, us, afs, bg])
        model.compile(optimizer=Adam(lr=learningRate),
                      loss={'pz': self.complex_loss(pzw), 'cz': self.complex_loss(czw), 'us': self.complex_loss(usw),
                            'afs': self.complex_loss(afsw), 'bg': self.complex_loss(bgw)},
                      metrics={'pz': self.dice_coef, 'cz': self.dice_coef, 'us': self.dice_coef,
                               'afs': self.dice_coef, 'bg': self.dice_coef})

        return model



    def train(self, train_imgs, train_gt, train_mask, val_imgs, val_gt_list, val_mask, bs, nr_epochs, csvFile, modelSaveFile, LR=5e-5, mask = False, bn = False, do = False, LRScheduling = False, EarlyStop=False, tbLogDir=''):

        csv_logger = CSVLogger(csvFile, append=True, separator=';')
        model_checkpoint = ModelCheckpoint(modelSaveFile, monitor='val_loss', save_best_only=True, verbose = 1, mode='min')
        tensorboard = TensorBoard(log_dir=LOG_DIR, write_graph=False, write_grads = True, histogram_freq=0, batch_size=5, write_images=False)
        earlyStopImprovement = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200, verbose=1, mode='min')
        LRDecay=ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, verbose=1, mode='min', min_lr=1e-8, epsilon=0.01)


        print('-' * 30)
        print('Loading train data...')
        print('-' * 30)

        print("Images Size:", train_imgs.shape)
        print("GT Size:", train_gt.shape)
        print("Mask Size:", train_mask.shape)

        print('-' * 30)
        print('Creating and compiling train...')
        print('-' * 30)

        model = self.get_net(1, LR, bn, do, mask)
        # plot_model(train, to_file='train.png')

        print('-' * 30)
        print('Fitting train...')
        print('-' * 30)

        cb=[csv_logger, model_checkpoint]
        if EarlyStop:
            cb.append(earlyStopImprovement)
        if LRScheduling:
            cb.append(LRDecay)
        cb.append(tensorboard)

        print('BATCH Size = ', bs)

        print('Callbacks: ',cb)
        params = {'dim': (32, 168, 168),
                  'batch_size': bs,
                  'n_classes': 5,
                  'n_channels': 1,
                  'shuffle': True,
                  'rotation': True}

        train_id_list = [str(i) for i in np.arange(0, train_imgs.shape[0])]
        training_generator = DataGenerator(train_imgs, train_gt, train_mask, train_id_list, bs, **params)

        val_id_list = [str(i) for i in np.arange(0, val_imgs.shape[0])]
        val_generator = ValDataGenerator(val_imgs, val_gt_list, val_mask, val_id_list, bs, **params)
        steps = 58/2
        #  history = train.fit(1, train_id_list, batch_size=bs, epochs=nr_epochs, verbose=1, validation_data=[val_imgs, val_gt_list], shuffle=True, callbacks=cb)
        history = model.fit_generator(generator=training_generator,
                                      steps_per_epoch= steps,
                                      validation_data=val_generator,
                                      use_multiprocessing=False,
                                      epochs =nr_epochs,
                                      callbacks = cb)
                            #workers=4)
        model.save(modelSaveFile[:-3] + '_final.h5')


        return history


def plot_loss(history, title):
    # "Loss"
    val = True
    result = history.history
    plt.plot(result['bg_loss'], 'b')
    if val:
        plt.plot(result['val_bg_loss'], 'b--')
    plt.plot(result['bg_loss'], 'b')
    plt.plot(result['pz_loss'], 'g')
    if val:
        plt.plot(result['val_pz_loss'], 'g--')
    plt.plot(result['cz_loss'], 'r')
    if val:
        plt.plot(result['val_cz_loss'], 'r--')
    plt.plot(result['us_loss'], 'y')
    if val:
        plt.plot(result['val_us_loss'], 'y--')
    plt.plot(result['afs_loss'], 'c')
    if val:
        plt.plot(result['val_afs_loss'], 'c--')
    plt.title(title)
    plt.ylabel('inv. Dice Coeff')
    axes = plt.gca()
    axes.set_ylim([-1.0, 0])
    axes.set_xlim([0, 600])
    plt.xlabel('epoch')
    plt.legend(['bg train', 'bg val', 'pz train', 'pz val', 'cz train', 'cz val', 'us train', 'us val', 'afs train',
                'afs val'], loc='upper right')
    plt.savefig(title + '.png')
    plt.clf()
    # plt.show()

def augmentData(array):
    size = array.shape
    bla = size[0] * 2
    dtype = array.dtype
    # print(dtype)
    array_new = np.zeros([bla, size[1], size[2], size[3], size[4]])

    for i in range(0, size[0]):
        # print(i)
        array_new[2 * i, :, :, :, :] = array[i, :, :, :, :]
        array_new[2 * i + 1, :, :, :, :] = np.flip(array[i, :, :, :, :], axis=2)

    return array_new


def load_data_and_train_model(train_imgs, train_gt, train_mask, epochs, learningRate, val_imgs, val_gt_list, val_mask):
    w_model = weighted_model()

    learningRate = float(learningRate)

    #nrChanels = 1
    name =  'augmented_x20_sfs16_dataGeneration_LR_' + str(learningRate)

    #id_list_train = [ str(i) for i in np.arange(0,train_imgs_no) ]

    his = w_model.train(train_imgs, train_gt, train_mask,
                        val_imgs, val_gt_list, val_mask,
                        bs=2, nr_epochs=int(epochs),
                        csvFile=name + '.csv',
                        modelSaveFile=name + '.h5',
                        LR=learningRate,
                        mask=False,
                        bn=True,
                        do=True,
                        LRScheduling=True,
                        EarlyStop=True,
                        tbLogDir=LOG_DIR + name)

    plot_loss(his, name)



def predict(val_imgs, val_gt_list, learningRate):


    nrChanels = 1

    name = 'augmented_x20_sfs16_dataGeneration_LR_' + str(learningRate)


    print(name)
    model = weighted_model()
    model = model.get_net(nrChanels, bn=True, do=False)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print('load_weights')
    model.load_weights(script_dir+'/data_78/'+name+'_final.h5')
    print('predict')
    out = model.predict([val_imgs], batch_size=1, verbose=1)

    print(model.evaluate([val_imgs], val_gt_list , batch_size=1, verbose=1))
    print(name)

    np.save(script_dir+'/predictions/predicted_'+name + '.npy', out)




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    img_rows = 168
    img_cols = 168
    img_depth = 168
    smooth = 1.
    train_imgs_no =58

    epochs = 300
    LR = 5e-5
    prediction = False


    #    LR decay
    train_imgs = np.load('/home/suhita/zonals/data/training/trainArray_imgs_fold1.npy')
    train_gt = np.load('/home/suhita/zonals/data/training/trainArray_GT_fold1.npy')
    train_mask = np.load('/home/suhita/zonals/data/training/distance_mask_train.npy')

    val_imgs = np.load('/home/suhita/zonals/data/validation/valArray_imgs_fold1.npy')
    val_gt = np.load('/home/suhita/zonals/data/validation/valArray_GT_fold1.npy')
    val_mask = np.load('/home/suhita/zonals/data/validation/distance_mask_val.npy')


    val_gt = val_gt.astype(int)
    val_gt_list = [val_gt[:, :, :, :, 0], val_gt[:, :, :, :, 1], val_gt[:, :, :, :, 2], val_gt[:, :, :, :, 3],
                   val_gt[:, :, :, :, 4]]

    if not prediction=='True':
        load_data_and_train_model(train_imgs, train_gt, train_mask, epochs, LR, val_imgs, val_gt, val_mask)
    else:
        np.save('imgs_test.npy', val_imgs)
        np.save('imgs_test_GT', val_gt)

        print('****** predict segmentations ******')
        predict(val_imgs, val_gt_list, LR)


    #load_data_and_train_model(gpu_id=0, weights=False)

