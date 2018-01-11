import sys
import os
import numpy as np
# from skimage.io import imsave, imshow
from scipy.misc import imsave, imshow
from keras.models import Sequential, Model
from keras.layers import Flatten, Input, Concatenate, Conv2DTranspose, Conv2D
from keras.layers import Reshape, Dense, Lambda, UpSampling2D
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from load_data import load_img, text2vector

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def GAN_0():
    dropout_rate = 0.25
    # Build Generative model ...
    nch = 128
    g_input = Input(shape=[zdim])
    H = Dense(nch * 32 * 32, init='glorot_normal')(g_input)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Reshape([32, 32, nch])(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(64, [3, 3], padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(32, [3, 3], padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(3, [1, 1], padding='same')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input, g_V)
    generator.summary()
    # Build Discriminative model ...
    d_input = Input(shape=(64, 64, 3))
    H = Conv2D(128, [5, 5], strides=(2, 2), padding='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(256, [5, 5], strides=(2, 2), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(512, [5, 5], strides=(2, 2), padding='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2, activation='softmax')(H)
    discriminator = Model(d_input, d_V)
    discriminator.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    discriminator.summary()

    # Freeze weights in the discriminator for stacked training

    make_trainable(discriminator, False)
    # Build stacked GAN model
    gan_input = Input(shape=[zdim])
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(optimizer=adam1, loss='categorical_crossentropy', metrics=['acc'])
    GAN.summary()

    return generator, discriminator, GAN


units = 64
maxepoch = 500000
vdim = 30
zdim = 100
batchsize = 32
adam = Adam(lr=0.0002)
adam1 = Adam(lr=0.0002)

if __name__ == '__main__':
    G, D, GAN = GAN_0()
    # exit()
    # train_D, train_G = WGAN_training()
    print('Load data...')

    # real_image = load_img('faces')
    # real_image = (real_image-0.5)/0.5
    # np.save('faces01.npy', real_image)
    # exit()

    real_image = np.load('faces01.npy')
    N = real_image.shape[0]
    # iteration = N//batchsize

    D_train_loss, G_train_loss = [], []
    for epoch in range(maxepoch):
        shuffle = np.random.permutation(N)
        batch = shuffle[:batchsize]
        I = real_image[batch]
        zzz = np.random.uniform(0,1,size=(batchsize, zdim))
        fake_I = G.predict(zzz)

        ### Update D
        # D.trainable = True
        img_input = np.concatenate((I, fake_I), axis=0)
        # loss_D = train_D([img_input, text_input])
        D_y = np.zeros([batchsize*2, 2])
        D_y[0:batchsize, 1] = 1
        D_y[batchsize:, 0] = 1
        loss_D , acc_D = D.train_on_batch(img_input, D_y)
        # loss_D , acc_D = D.train_on_batch(img_input[batchsize:], D_y[batchsize:])
        D_train_loss.append(loss_D)
        ### Update G
        # D.trainable = False
        GAN_input_z = np.random.uniform(0,1,size=(batchsize, zdim))
        # loss_G = train_G([GAN_input_vz, GAN_input_v])
        GAN_y = np.zeros([batchsize, 2])
        GAN_y[:,1] = 1
        loss_G, acc_G = GAN.train_on_batch(GAN_input_z, GAN_y)
        G_train_loss.append(loss_G)

        ### Save Weights and Print loss
        print('epoch:', epoch, 'loss_D:', loss_D, 'acc_D:', acc_D,
              'loss_G:',loss_G, 'acc_G:', acc_G,)
        if (epoch+1)%10 ==0:
            G.save_weights('WGAN_G_weights.h5')
            D.save_weights('WGAN_D_weights.h5')
            np.save('z2img_Gloss.npy', G_train_loss)
            np.save('z2img_Dloss.npy', D_train_loss)
        if (epoch + 1) % 200 == 0:
            for s in range(3):
                imsave('img/'+str(epoch) + '_' + str(shuffle[s]) + '.jpg', fake_I[s])

